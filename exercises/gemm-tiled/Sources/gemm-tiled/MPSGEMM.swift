import Dispatch
import Foundation
import Metal
import MetalPerformanceShaders

private struct MPSDispatchTiming {
    let wallMs: Double
    let deviceMs: Double?
}

private struct MPSMatrixLayout {
    let rows: Int
    let columns: Int
    let rowBytes: Int

    init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
        self.rowBytes = MPSMatrixDescriptor.rowBytes(forColumns: columns, dataType: .float32)
    }

    var rowStrideElements: Int {
        rowBytes / MemoryLayout<Float>.stride
    }

    var bufferLength: Int {
        rowBytes * rows
    }

    var descriptor: MPSMatrixDescriptor {
        MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: rowBytes, dataType: .float32)
    }
}

struct MPSGEMMRunner: GEMMRunner {
    let name = "MPSMatrixMultiplication"

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.unsupportedPlatform("Metal is unavailable on this machine")
        }
        guard MPSSupportsMTLDevice(device) else {
            throw BenchmarkError.unsupportedPlatform("Metal Performance Shaders is unavailable on this machine")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command queue for MPS")
        }

        self.device = device
        self.commandQueue = commandQueue
    }

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        let aLayout = MPSMatrixLayout(rows: problem.m, columns: problem.k)
        let bLayout = MPSMatrixLayout(rows: problem.k, columns: problem.n)
        let cLayout = MPSMatrixLayout(rows: problem.m, columns: problem.n)

        let multiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: problem.m,
            resultColumns: problem.n,
            interiorColumns: problem.k,
            alpha: 1.0,
            beta: 0.0
        )

        let aStagingBuffer = try makeSharedBuffer(length: aLayout.bufferLength, errorMessage: "Could not allocate the MPS staging buffer for A")
        let bStagingBuffer = try makeSharedBuffer(length: bLayout.bufferLength, errorMessage: "Could not allocate the MPS staging buffer for B")
        let aBuffer = try makePrivateBuffer(length: aLayout.bufferLength, errorMessage: "Could not allocate the private MPS buffer for A")
        let bBuffer = try makePrivateBuffer(length: bLayout.bufferLength, errorMessage: "Could not allocate the private MPS buffer for B")
        let cBuffer = try makePrivateBuffer(length: cLayout.bufferLength, errorMessage: "Could not allocate the private MPS buffer for C")
        let cReadbackBuffer = try makeSharedBuffer(length: cLayout.bufferLength, errorMessage: "Could not allocate the MPS readback buffer for C")

        writeMatrix(a.elements, into: aStagingBuffer, layout: aLayout)
        writeMatrix(b.elements, into: bStagingBuffer, layout: bLayout)

        try copyBuffer(aStagingBuffer, to: aBuffer)
        try copyBuffer(bStagingBuffer, to: bBuffer)

        let aMatrix = MPSMatrix(buffer: aBuffer, descriptor: aLayout.descriptor)
        let bMatrix = MPSMatrix(buffer: bBuffer, descriptor: bLayout.descriptor)
        let cMatrix = MPSMatrix(buffer: cBuffer, descriptor: cLayout.descriptor)

        var wallTimings = [Double]()
        var deviceTimings = [Double]()

        for iteration in 0..<(warmupIterations + measuredIterations) {
            let timing = try dispatchMultiply(
                multiplication: multiplication,
                leftMatrix: aMatrix,
                rightMatrix: bMatrix,
                resultMatrix: cMatrix
            )

            if iteration >= warmupIterations {
                wallTimings.append(timing.wallMs)
                if let deviceMs = timing.deviceMs {
                    deviceTimings.append(deviceMs)
                }
            }
        }

        try copyBuffer(cBuffer, to: cReadbackBuffer)
        let output = readMatrix(from: cReadbackBuffer, layout: cLayout)

        return RawBenchmarkRun(
            output: Matrix(rows: problem.m, cols: problem.n, elements: output),
            wallAverageMs: wallTimings.average,
            wallBestMs: wallTimings.min() ?? 0,
            deviceAverageMs: deviceTimings.isEmpty ? nil : deviceTimings.average,
            deviceBestMs: deviceTimings.min()
        )
    }

    private func dispatchMultiply(
        multiplication: MPSMatrixMultiplication,
        leftMatrix: MPSMatrix,
        rightMatrix: MPSMatrix,
        resultMatrix: MPSMatrix
    ) throws -> MPSDispatchTiming {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command buffer for MPS")
        }

        multiplication.encode(commandBuffer: commandBuffer, leftMatrix: leftMatrix, rightMatrix: rightMatrix, resultMatrix: resultMatrix)

        let wallStart = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallElapsedMs = Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1_000_000.0

        if let error = commandBuffer.error {
            throw error
        }

        return MPSDispatchTiming(
            wallMs: wallElapsedMs,
            deviceMs: gpuElapsedMs(for: commandBuffer)
        )
    }

    private func writeMatrix(_ elements: [Float], into buffer: MTLBuffer, layout: MPSMatrixLayout) {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: layout.rows * layout.rowStrideElements)
        for row in 0..<layout.rows {
            let destinationBase = row * layout.rowStrideElements
            let sourceBase = row * layout.columns
            for column in 0..<layout.columns {
                pointer[destinationBase + column] = elements[sourceBase + column]
            }
            if layout.rowStrideElements > layout.columns {
                let paddingStart = destinationBase + layout.columns
                let paddingCount = layout.rowStrideElements - layout.columns
                for offset in 0..<paddingCount {
                    pointer[paddingStart + offset] = 0
                }
            }
        }
    }

    private func readMatrix(from buffer: MTLBuffer, layout: MPSMatrixLayout) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: layout.rows * layout.rowStrideElements)
        var elements = [Float](repeating: 0, count: layout.rows * layout.columns)

        for row in 0..<layout.rows {
            let sourceBase = row * layout.rowStrideElements
            let destinationBase = row * layout.columns
            for column in 0..<layout.columns {
                elements[destinationBase + column] = pointer[sourceBase + column]
            }
        }

        return elements
    }

    private func copyBuffer(_ source: MTLBuffer, to destination: MTLBuffer) throws {
        guard source.length <= destination.length else {
            throw BenchmarkError.runtimeFailure("MPS buffer copy exceeds destination size")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal command buffer for MPS blit")
        }
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw BenchmarkError.runtimeFailure("Could not create a Metal blit encoder for MPS")
        }

        encoder.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: source.length)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }
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

    private func gpuElapsedMs(for commandBuffer: MTLCommandBuffer) -> Double? {
        let elapsed = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0
        return elapsed > 0 ? elapsed : nil
    }
}
