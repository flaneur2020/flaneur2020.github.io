import Dispatch
import Foundation
import Metal

private struct GEMMUniforms {
    var m: UInt32
    var n: UInt32
    var k: UInt32
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

struct MetalKernelConfiguration {
    let name: String
    let functionName: String
    let threadgroupWidth: Int
    let threadgroupHeight: Int
    let outputTileWidth: Int
    let outputTileHeight: Int
    let aOperandLayout: MetalAOperandLayout
    let bOperandLayout: MetalBOperandLayout
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

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        let preparedAElements = prepareAOperandElements(from: a, problem: problem)
        let preparedBElements = prepareBOperandElements(from: b, problem: problem)
        let aBuffer = try makeBuffer(copying: preparedAElements)
        let bBuffer = try makeBuffer(copying: preparedBElements)
        let outputLength = problem.outputElementCount * MemoryLayout<Float>.stride
        guard let cBuffer = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
            throw BenchmarkError.runtimeFailure("Could not allocate the output Metal buffer")
        }

        var uniforms = GEMMUniforms(
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

        var timings = [Double]()

        for iteration in 0..<(warmupIterations + measuredIterations) {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw BenchmarkError.runtimeFailure("Could not create a Metal command buffer")
            }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw BenchmarkError.runtimeFailure("Could not create a Metal compute encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&uniforms, length: MemoryLayout<GEMMUniforms>.stride, index: 3)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()

            let wallStart = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let wallElapsedMs = Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1_000_000.0

            if let error = commandBuffer.error {
                throw error
            }

            if iteration >= warmupIterations {
                timings.append(gpuElapsedMs(for: commandBuffer, fallbackMs: wallElapsedMs))
            }
        }

        return RawBenchmarkRun(
            output: Matrix(rows: problem.m, cols: problem.n, elements: readFloats(from: cBuffer, count: problem.outputElementCount)),
            averageMs: timings.average,
            bestMs: timings.min() ?? 0
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

    private func makeBuffer(copying values: [Float]) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw BenchmarkError.runtimeFailure("Could not allocate a Metal input buffer")
        }

        values.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                buffer.contents().copyMemory(from: baseAddress, byteCount: length)
            }
        }
        return buffer
    }

    private func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func gpuElapsedMs(for commandBuffer: MTLCommandBuffer, fallbackMs: Double) -> Double {
        let elapsed = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0
        return elapsed > 0 ? elapsed : fallbackMs
    }

    private static func loadShaderSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "GEMMShaders", withExtension: "metal") else {
            throw BenchmarkError.runtimeFailure("Could not locate Resources/GEMMShaders.metal")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
