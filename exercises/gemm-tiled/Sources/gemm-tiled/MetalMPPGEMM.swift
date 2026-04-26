import Dispatch
import Foundation
import Metal
import Metal4Interop

private struct MPPDispatchTiming {
    let wallMs: Double
    let deviceMs: Double?
}

struct MPPKernelConfiguration {
    let name: String
    let functionName: String
    let threadgroupWidth: Int
    let maxTileWidth: Int
    let maxTileHeight: Int
    let maxSupportedK: Int
}

let mppRunnerConfigurations = [
    MPPKernelConfiguration(
        name: "Metal cooperative matrix 64x32 (k<64)",
        functionName: "mpp_matmul_cooperative_f32_64x32",
        threadgroupWidth: 128,
        maxTileWidth: 32,
        maxTileHeight: 64,
        maxSupportedK: 63
    ),
    MPPKernelConfiguration(
        name: "Metal cooperative matrix 32x32 (k<64)",
        functionName: "mpp_matmul_cooperative_f32_32x32",
        threadgroupWidth: 64,
        maxTileWidth: 32,
        maxTileHeight: 32,
        maxSupportedK: 63
    ),
]

@available(macOS 26.0, *)
struct MetalMPPGEMMRunner: GEMMRunner {
    let name: String

    private let configuration: MPPKernelConfiguration
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let commandAllocator: any MTL4CommandAllocator
    private let commandQueue: any MTL4CommandQueue
    private let argumentTableBindingCount: Int
    private let aBindingIndex: Int
    private let bBindingIndex: Int
    private let cBindingIndex: Int
    private let threadsPerThreadgroup: MTLSize

    init(configuration: MPPKernelConfiguration) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.unsupportedPlatform("Metal is unavailable on this machine")
        }

        let compileOptions = MTLCompileOptions()
        compileOptions.languageVersion = .version4_0

        let source = try Self.loadShaderSource()
        let library = try device.makeLibrary(source: source, options: compileOptions)
        guard let function = library.makeFunction(name: configuration.functionName) else {
            throw BenchmarkError.runtimeFailure("Could not find \(configuration.functionName) in MPPShaders.metal")
        }

        let threadsPerThreadgroup = MTLSize(width: configuration.threadgroupWidth, height: 1, depth: 1)

        let pipelineDescriptor = MTLComputePipelineDescriptor()
        pipelineDescriptor.computeFunction = function
        pipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        pipelineDescriptor.requiredThreadsPerThreadgroup = threadsPerThreadgroup

        var reflection: MTLAutoreleasedComputePipelineReflection?
        let pipeline = try device.makeComputePipelineState(
            descriptor: pipelineDescriptor,
            options: .bindingInfo,
            reflection: &reflection
        )
        guard let reflection else {
            throw BenchmarkError.runtimeFailure("Could not reflect Metal MPP pipeline bindings")
        }

        guard let commandAllocator = device.makeCommandAllocator() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal 4 command allocator")
        }
        guard let commandQueue = device.makeMTL4CommandQueue() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal 4 command queue")
        }

        let tensorBindings = reflection.bindings.filter { $0.type == .tensor }
        guard let aBinding = tensorBindings.first(where: { $0.name == "aTensor" }),
              let bBinding = tensorBindings.first(where: { $0.name == "bTensor" }),
              let cBinding = tensorBindings.first(where: { $0.name == "cTensor" }) else {
            throw BenchmarkError.runtimeFailure("Could not resolve tensor bindings for Metal MPP pipeline")
        }

        self.name = configuration.name
        self.configuration = configuration
        self.device = device
        self.pipeline = pipeline
        self.commandAllocator = commandAllocator
        self.commandQueue = commandQueue
        self.argumentTableBindingCount = (tensorBindings.map(\.index).max() ?? 0) + 1
        self.aBindingIndex = aBinding.index
        self.bBindingIndex = bBinding.index
        self.cBindingIndex = cBinding.index
        self.threadsPerThreadgroup = threadsPerThreadgroup
    }

    func supports(problem: GEMMProblem) -> Bool {
        problem.m <= configuration.maxTileHeight && problem.n <= configuration.maxTileWidth && problem.k <= configuration.maxSupportedK
    }

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        let aBuffer = try makeSharedBuffer(copying: a.elements, errorMessage: "Could not allocate \(name) input buffer A")
        let bBuffer = try makeSharedBuffer(copying: b.elements, errorMessage: "Could not allocate \(name) input buffer B")
        let cBuffer = try makeSharedBuffer(length: problem.outputElementCount * MemoryLayout<Float>.stride, errorMessage: "Could not allocate \(name) output buffer C")

        let aTensor = try makeTensor(from: aBuffer, logicalWidth: problem.k, logicalHeight: problem.m, rowStride: problem.k)
        let bTensor = try makeTensor(from: bBuffer, logicalWidth: problem.n, logicalHeight: problem.k, rowStride: problem.n)
        let cTensor = try makeTensor(from: cBuffer, logicalWidth: problem.n, logicalHeight: problem.m, rowStride: problem.n)
        let argumentTable = try makeArgumentTable(aTensor: aTensor, bTensor: bTensor, cTensor: cTensor)

        var wallTimings = [Double]()
        var deviceTimings = [Double]()

        for iteration in 0..<(warmupIterations + measuredIterations) {
            memset(cBuffer.contents(), 0, cBuffer.length)
            let timing = try dispatch(argumentTable: argumentTable)

            if iteration >= warmupIterations {
                wallTimings.append(timing.wallMs)
                if let deviceMs = timing.deviceMs {
                    deviceTimings.append(deviceMs)
                }
            }
        }

        let output = readFloats(from: cBuffer, count: problem.outputElementCount)
        return RawBenchmarkRun(
            output: Matrix(rows: problem.m, cols: problem.n, elements: output),
            wallAverageMs: wallTimings.average,
            wallBestMs: wallTimings.min() ?? 0,
            deviceAverageMs: deviceTimings.isEmpty ? nil : deviceTimings.average,
            deviceBestMs: deviceTimings.min()
        )
    }

    private func dispatch(argumentTable: any MTL4ArgumentTable) throws -> MPPDispatchTiming {
        guard let commandBuffer = device.makeCommandBuffer() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal 4 command buffer")
        }
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        defer { commandAllocator.reset() }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal 4 compute command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setArgumentTable(argumentTable)

        let wallStart = DispatchTime.now().uptimeNanoseconds
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        var gpuStartTime = 0.0
        var gpuEndTime = 0.0
        try Metal4Interop.commit(
            commandBuffer,
            on: commandQueue,
            gpuStartTime: &gpuStartTime,
            gpuEndTime: &gpuEndTime
        )
        let wallElapsedMs = Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1_000_000.0
        let gpuElapsedMs = gpuEndTime > gpuStartTime ? (gpuEndTime - gpuStartTime) * 1_000.0 : nil

        return MPPDispatchTiming(wallMs: wallElapsedMs, deviceMs: gpuElapsedMs)
    }

    private func makeArgumentTable(
        aTensor: any MTLTensor,
        bTensor: any MTLTensor,
        cTensor: any MTLTensor
    ) throws -> any MTL4ArgumentTable {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = argumentTableBindingCount
        descriptor.initializeBindings = true

        let argumentTable: any MTL4ArgumentTable
        do {
            argumentTable = try device.makeArgumentTable(descriptor: descriptor)
        } catch {
            throw BenchmarkError.runtimeFailure("Could not create a Metal 4 argument table: \(error.localizedDescription)")
        }

        argumentTable.setResource(aTensor.gpuResourceID, bufferIndex: aBindingIndex)
        argumentTable.setResource(bTensor.gpuResourceID, bufferIndex: bBindingIndex)
        argumentTable.setResource(cTensor.gpuResourceID, bufferIndex: cBindingIndex)
        return argumentTable
    }

    private func makeSharedBuffer(copying values: [Float], errorMessage: String) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw BenchmarkError.runtimeFailure(errorMessage)
        }

        values.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                buffer.contents().copyMemory(from: baseAddress, byteCount: length)
            }
        }
        return buffer
    }

    private func makeSharedBuffer(length: Int, errorMessage: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw BenchmarkError.runtimeFailure(errorMessage)
        }
        return buffer
    }

    private func makeTensor(
        from buffer: MTLBuffer,
        logicalWidth: Int,
        logicalHeight: Int,
        rowStride: Int
    ) throws -> any MTLTensor {
        let descriptor = MTLTensorDescriptor()
        descriptor.dataType = .float32
        descriptor.usage = .compute
        descriptor.dimensions = try makeExtents([logicalWidth, logicalHeight])
        descriptor.strides = try makeExtents([1, rowStride])
        return try buffer.makeTensor(descriptor: descriptor, offset: 0)
    }

    private func makeExtents(_ values: [Int]) throws -> MTLTensorExtents {
        let extents = values.withUnsafeBufferPointer { pointer in
            MTLTensorExtents(__rank: values.count, values: pointer.baseAddress)
        }
        guard let extents else {
            throw BenchmarkError.runtimeFailure("Could not create MTLTensorExtents")
        }
        return extents
    }

    private func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func loadShaderSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "MPPShaders", withExtension: "metal") else {
            throw BenchmarkError.runtimeFailure("Could not locate Resources/MPPShaders.metal")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
