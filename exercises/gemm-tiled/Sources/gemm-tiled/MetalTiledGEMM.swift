import Dispatch
import Foundation
import Metal

private struct GEMMUniforms {
    var m: UInt32
    var n: UInt32
    var k: UInt32
}

struct MetalKernelConfiguration {
    let name: String
    let functionName: String
    let threadgroupWidth: Int
    let threadgroupHeight: Int
    let outputTileWidth: Int
    let outputTileHeight: Int
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
        let aBuffer = try makeBuffer(copying: a.elements)
        let bBuffer = try makeBuffer(copying: b.elements)
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
