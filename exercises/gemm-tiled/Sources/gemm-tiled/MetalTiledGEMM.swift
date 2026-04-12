import Dispatch
import Foundation
import Metal

private struct GEMMUniforms {
    var m: UInt32
    var n: UInt32
    var k: UInt32
}

private struct MetalDispatchTiming {
    let wallMs: Double
    let deviceMs: Double?
}

enum MetalAOperandLayout {
    case rowMajor
    case packedVectorized(blockM: Int, blockK: Int, vectorHeight: Int)
}

enum MetalBOperandLayout {
    case rowMajor
    case packedSwizzled(blockK: Int, blockN: Int, swizzleGroup: Int)
    case packedVectorized(blockK: Int, blockN: Int, vectorWidth: Int)
    case packedVectorizedSwizzled(blockK: Int, blockN: Int, vectorWidth: Int, vectorSwizzleGroup: Int)
}

enum MetalBufferMode {
    case shared
    case privateStaged
}

struct MetalKernelConfiguration {
    let name: String
    let functionName: String
    let threadgroupWidth: Int
    let threadgroupHeight: Int
    let outputTileWidth: Int
    let outputTileHeight: Int
    let aOperandLayout: MetalAOperandLayout
    let bOperandLayout: MetalBOperandLayout
    let bufferMode: MetalBufferMode
    let requiresAlignedProblem: Bool
    let requiredKAlignment: Int

    init(
        name: String,
        functionName: String,
        threadgroupWidth: Int,
        threadgroupHeight: Int,
        outputTileWidth: Int,
        outputTileHeight: Int,
        aOperandLayout: MetalAOperandLayout,
        bOperandLayout: MetalBOperandLayout,
        bufferMode: MetalBufferMode = .shared,
        requiresAlignedProblem: Bool = false,
        requiredKAlignment: Int = 1
    ) {
        self.name = name
        self.functionName = functionName
        self.threadgroupWidth = threadgroupWidth
        self.threadgroupHeight = threadgroupHeight
        self.outputTileWidth = outputTileWidth
        self.outputTileHeight = outputTileHeight
        self.aOperandLayout = aOperandLayout
        self.bOperandLayout = bOperandLayout
        self.bufferMode = bufferMode
        self.requiresAlignedProblem = requiresAlignedProblem
        self.requiredKAlignment = requiredKAlignment
    }
}

struct MetalTiledGEMMRunner: GEMMRunner {
    let name: String

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let configuration: MetalKernelConfiguration

    init(configuration: MetalKernelConfiguration) throws {
        self.name = configuration.name
        self.configuration = configuration

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.unsupportedPlatform("Metal is unavailable on this machine")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command queue")
        }

        let source = try Self.loadShaderSource()
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: configuration.functionName) else {
            throw BenchmarkError.runtimeFailure("Could not find \(configuration.functionName) in GEMMShaders.metal")
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        let threadCount = configuration.threadgroupWidth * configuration.threadgroupHeight
        guard threadCount <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw BenchmarkError.runtimeFailure("The \(configuration.name) threadgroup exceeds the device limit")
        }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
    }

    func supports(problem: GEMMProblem) -> Bool {
        guard configuration.requiresAlignedProblem else {
            return true
        }

        return problem.m % configuration.outputTileHeight == 0 &&
            problem.n % configuration.outputTileWidth == 0 &&
            problem.k % configuration.requiredKAlignment == 0
    }

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        let preparedAElements = prepareAOperandElements(from: a, problem: problem)
        let preparedBElements = prepareBOperandElements(from: b, problem: problem)
        let outputLength = problem.outputElementCount * MemoryLayout<Float>.stride
        let uniforms = GEMMUniforms(
            m: UInt32(problem.m),
            n: UInt32(problem.n),
            k: UInt32(problem.k)
        )

        let threadsPerThreadgroup = MTLSize(
            width: configuration.threadgroupWidth,
            height: configuration.threadgroupHeight,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (problem.n + configuration.outputTileWidth - 1) / configuration.outputTileWidth,
            height: (problem.m + configuration.outputTileHeight - 1) / configuration.outputTileHeight,
            depth: 1
        )

        let timings: [MetalDispatchTiming]
        let outputBuffer: MTLBuffer

        switch configuration.bufferMode {
        case .shared:
            let aBuffer = try makeSharedBuffer(copying: preparedAElements, errorMessage: "Could not allocate a Metal input buffer")
            let bBuffer = try makeSharedBuffer(copying: preparedBElements, errorMessage: "Could not allocate a Metal input buffer")
            let cBuffer = try makeSharedBuffer(length: outputLength, errorMessage: "Could not allocate the output Metal buffer")
            timings = try runBenchmarkIterations(
                aBuffer: aBuffer,
                bBuffer: bBuffer,
                cBuffer: cBuffer,
                uniforms: uniforms,
                threadgroups: threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup,
                warmupIterations: warmupIterations,
                measuredIterations: measuredIterations
            )
            outputBuffer = cBuffer

        case .privateStaged:
            let aStagingBuffer = try makeSharedBuffer(copying: preparedAElements, errorMessage: "Could not allocate the Metal staging buffer for A")
            let bStagingBuffer = try makeSharedBuffer(copying: preparedBElements, errorMessage: "Could not allocate the Metal staging buffer for B")
            let aBuffer = try makePrivateBuffer(length: aStagingBuffer.length, errorMessage: "Could not allocate the private Metal buffer for A")
            let bBuffer = try makePrivateBuffer(length: bStagingBuffer.length, errorMessage: "Could not allocate the private Metal buffer for B")
            let cBuffer = try makePrivateBuffer(length: outputLength, errorMessage: "Could not allocate the private Metal buffer for C")
            let cReadbackBuffer = try makeSharedBuffer(length: outputLength, errorMessage: "Could not allocate the Metal readback buffer for C")

            try copyBuffer(aStagingBuffer, to: aBuffer)
            try copyBuffer(bStagingBuffer, to: bBuffer)

            timings = try runBenchmarkIterations(
                aBuffer: aBuffer,
                bBuffer: bBuffer,
                cBuffer: cBuffer,
                uniforms: uniforms,
                threadgroups: threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup,
                warmupIterations: warmupIterations,
                measuredIterations: measuredIterations
            )

            try copyBuffer(cBuffer, to: cReadbackBuffer)
            outputBuffer = cReadbackBuffer
        }

        let wallTimings = timings.map { $0.wallMs }
        let deviceTimings = timings.compactMap { $0.deviceMs }

        return RawBenchmarkRun(
            output: Matrix(rows: problem.m, cols: problem.n, elements: readFloats(from: outputBuffer, count: problem.outputElementCount)),
            wallAverageMs: wallTimings.average,
            wallBestMs: wallTimings.min() ?? 0,
            deviceAverageMs: deviceTimings.isEmpty ? nil : deviceTimings.average,
            deviceBestMs: deviceTimings.min()
        )
    }

    private func prepareAOperandElements(from a: Matrix, problem: GEMMProblem) -> [Float] {
        switch configuration.aOperandLayout {
        case .rowMajor:
            return a.elements
        case .packedVectorized(let blockM, let blockK, let vectorHeight):
            return packVectorizedAOperand(
                elements: a.elements,
                problem: problem,
                blockM: blockM,
                blockK: blockK,
                vectorHeight: vectorHeight
            )
        }
    }

    private func prepareBOperandElements(from b: Matrix, problem: GEMMProblem) -> [Float] {
        switch configuration.bOperandLayout {
        case .rowMajor:
            return b.elements
        case .packedSwizzled(let blockK, let blockN, let swizzleGroup):
            return packSwizzledBOperand(
                elements: b.elements,
                problem: problem,
                blockK: blockK,
                blockN: blockN,
                swizzleGroup: swizzleGroup
            )
        case .packedVectorized(let blockK, let blockN, let vectorWidth):
            return packVectorizedBOperand(
                elements: b.elements,
                problem: problem,
                blockK: blockK,
                blockN: blockN,
                vectorWidth: vectorWidth
            )
        case .packedVectorizedSwizzled(let blockK, let blockN, let vectorWidth, let vectorSwizzleGroup):
            return packVectorizedSwizzledBOperand(
                elements: b.elements,
                problem: problem,
                blockK: blockK,
                blockN: blockN,
                vectorWidth: vectorWidth,
                vectorSwizzleGroup: vectorSwizzleGroup
            )
        }
    }

    private func packVectorizedAOperand(
        elements: [Float],
        problem: GEMMProblem,
        blockM: Int,
        blockK: Int,
        vectorHeight: Int
    ) -> [Float] {
        precondition(blockM % vectorHeight == 0)

        let mTileCount = (problem.m + blockM - 1) / blockM
        let kTileCount = (problem.k + blockK - 1) / blockK
        let vectorsPerInner = blockM / vectorHeight
        let tileElementCount = blockM * blockK
        var packed = [Float](repeating: 0, count: mTileCount * kTileCount * tileElementCount)

        for mTile in 0..<mTileCount {
            for kTile in 0..<kTileCount {
                let tileBase = (mTile * kTileCount + kTile) * tileElementCount
                for inner in 0..<blockK {
                    let sourceColumn = kTile * blockK + inner
                    for vectorIndex in 0..<vectorsPerInner {
                        for lane in 0..<vectorHeight {
                            let sourceRow = mTile * blockM + vectorIndex * vectorHeight + lane
                            let destinationIndex = tileBase + inner * blockM + vectorIndex * vectorHeight + lane
                            if sourceRow < problem.m && sourceColumn < problem.k {
                                packed[destinationIndex] = elements[sourceRow * problem.k + sourceColumn]
                            }
                        }
                    }
                }
            }
        }

        return packed
    }

    private func packSwizzledBOperand(
        elements: [Float],
        problem: GEMMProblem,
        blockK: Int,
        blockN: Int,
        swizzleGroup: Int
    ) -> [Float] {
        let kTileCount = (problem.k + blockK - 1) / blockK
        let nTileCount = (problem.n + blockN - 1) / blockN
        let tileElementCount = blockK * blockN
        var packed = [Float](repeating: 0, count: nTileCount * kTileCount * tileElementCount)

        for nTile in 0..<nTileCount {
            for kTile in 0..<kTileCount {
                let tileBase = (nTile * kTileCount + kTile) * tileElementCount
                for inner in 0..<blockK {
                    let sourceRow = kTile * blockK + inner
                    for column in 0..<blockN {
                        let sourceColumn = nTile * blockN + column
                        let swizzledColumn = swizzledColumnIndex(
                            column: column,
                            inner: inner,
                            swizzleGroup: swizzleGroup
                        )
                        let value: Float
                        if sourceRow < problem.k && sourceColumn < problem.n {
                            value = elements[sourceRow * problem.n + sourceColumn]
                        } else {
                            value = 0
                        }
                        packed[tileBase + inner * blockN + swizzledColumn] = value
                    }
                }
            }
        }

        return packed
    }

    private func packVectorizedBOperand(
        elements: [Float],
        problem: GEMMProblem,
        blockK: Int,
        blockN: Int,
        vectorWidth: Int
    ) -> [Float] {
        precondition(blockN % vectorWidth == 0)

        let kTileCount = (problem.k + blockK - 1) / blockK
        let nTileCount = (problem.n + blockN - 1) / blockN
        let tileElementCount = blockK * blockN
        var packed = [Float](repeating: 0, count: nTileCount * kTileCount * tileElementCount)

        for nTile in 0..<nTileCount {
            for kTile in 0..<kTileCount {
                let tileBase = (nTile * kTileCount + kTile) * tileElementCount
                for inner in 0..<blockK {
                    let sourceRow = kTile * blockK + inner
                    for vectorIndex in 0..<(blockN / vectorWidth) {
                        for lane in 0..<vectorWidth {
                            let sourceColumn = nTile * blockN + vectorIndex * vectorWidth + lane
                            let destinationIndex = tileBase + inner * blockN + vectorIndex * vectorWidth + lane
                            if sourceRow < problem.k && sourceColumn < problem.n {
                                packed[destinationIndex] = elements[sourceRow * problem.n + sourceColumn]
                            }
                        }
                    }
                }
            }
        }

        return packed
    }

    private func packVectorizedSwizzledBOperand(
        elements: [Float],
        problem: GEMMProblem,
        blockK: Int,
        blockN: Int,
        vectorWidth: Int,
        vectorSwizzleGroup: Int
    ) -> [Float] {
        precondition(blockN % vectorWidth == 0)

        let vectorsPerRow = blockN / vectorWidth
        precondition(vectorsPerRow % vectorSwizzleGroup == 0)

        let kTileCount = (problem.k + blockK - 1) / blockK
        let nTileCount = (problem.n + blockN - 1) / blockN
        let tileElementCount = blockK * blockN
        var packed = [Float](repeating: 0, count: nTileCount * kTileCount * tileElementCount)

        for nTile in 0..<nTileCount {
            for kTile in 0..<kTileCount {
                let tileBase = (nTile * kTileCount + kTile) * tileElementCount
                for inner in 0..<blockK {
                    let sourceRow = kTile * blockK + inner
                    for vectorIndex in 0..<vectorsPerRow {
                        let swizzledVectorIndex = swizzledVectorIndex(
                            vectorIndex: vectorIndex,
                            inner: inner,
                            vectorSwizzleGroup: vectorSwizzleGroup
                        )
                        for lane in 0..<vectorWidth {
                            let sourceColumn = nTile * blockN + vectorIndex * vectorWidth + lane
                            let destinationIndex = tileBase + inner * blockN + swizzledVectorIndex * vectorWidth + lane
                            if sourceRow < problem.k && sourceColumn < problem.n {
                                packed[destinationIndex] = elements[sourceRow * problem.n + sourceColumn]
                            }
                        }
                    }
                }
            }
        }

        return packed
    }

    private func swizzledColumnIndex(column: Int, inner: Int, swizzleGroup: Int) -> Int {
        let groupBase = column / swizzleGroup * swizzleGroup
        let offset = column % swizzleGroup
        let swizzledOffset = offset ^ (inner % swizzleGroup)
        return groupBase + swizzledOffset
    }

    private func swizzledVectorIndex(vectorIndex: Int, inner: Int, vectorSwizzleGroup: Int) -> Int {
        let groupBase = vectorIndex / vectorSwizzleGroup * vectorSwizzleGroup
        let offset = vectorIndex % vectorSwizzleGroup
        let swizzledOffset = offset ^ (inner % vectorSwizzleGroup)
        return groupBase + swizzledOffset
    }

    private func runBenchmarkIterations(
        aBuffer: MTLBuffer,
        bBuffer: MTLBuffer,
        cBuffer: MTLBuffer,
        uniforms: GEMMUniforms,
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> [MetalDispatchTiming] {
        var timings = [MetalDispatchTiming]()

        for iteration in 0..<(warmupIterations + measuredIterations) {
            let timing = try dispatchKernel(
                aBuffer: aBuffer,
                bBuffer: bBuffer,
                cBuffer: cBuffer,
                uniforms: uniforms,
                threadgroups: threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup
            )

            if iteration >= warmupIterations {
                timings.append(timing)
            }
        }

        return timings
    }

    private func dispatchKernel(
        aBuffer: MTLBuffer,
        bBuffer: MTLBuffer,
        cBuffer: MTLBuffer,
        uniforms: GEMMUniforms,
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) throws -> MetalDispatchTiming {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal compute encoder")
        }

        var mutableUniforms = uniforms
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&mutableUniforms, length: MemoryLayout<GEMMUniforms>.stride, index: 3)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        let wallStart = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallElapsedMs = Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1_000_000.0

        if let error = commandBuffer.error {
            throw error
        }

        return MetalDispatchTiming(
            wallMs: wallElapsedMs,
            deviceMs: gpuElapsedMs(for: commandBuffer)
        )
    }

    private func copyBuffer(_ source: MTLBuffer, to destination: MTLBuffer) throws {
        guard source.length <= destination.length else {
            throw BenchmarkError.runtimeFailure("Metal buffer copy exceeds destination size")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command buffer for blit")
        }
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal blit encoder")
        }

        encoder.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: source.length)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }
    }

    private func makeSharedBuffer(copying values: [Float], errorMessage: String) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<Float>.stride
        let buffer = try makeSharedBuffer(length: length, errorMessage: errorMessage)

        values.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                buffer.contents().copyMemory(from: baseAddress, byteCount: length)
            }
        }

        return buffer
    }

    private func makeSharedBuffer(length: Int, errorMessage: String) throws -> MTLBuffer {
        try makeBuffer(length: length, options: .storageModeShared, errorMessage: errorMessage)
    }

    private func makePrivateBuffer(length: Int, errorMessage: String) throws -> MTLBuffer {
        try makeBuffer(length: length, options: .storageModePrivate, errorMessage: errorMessage)
    }

    private func makeBuffer(length: Int, options: MTLResourceOptions, errorMessage: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: options) else {
            throw BenchmarkError.runtimeFailure(errorMessage)
        }
        return buffer
    }

    private func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func gpuElapsedMs(for commandBuffer: MTLCommandBuffer) -> Double? {
        let elapsed = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0
        return elapsed > 0 ? elapsed : nil
    }

    private static func loadShaderSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "GEMMShaders", withExtension: "metal") else {
            throw BenchmarkError.runtimeFailure("Could not locate Resources/GEMMShaders.metal")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
