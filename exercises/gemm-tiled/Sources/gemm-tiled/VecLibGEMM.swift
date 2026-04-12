import Accelerate
import Foundation

struct VecLibGEMMRunner: GEMMRunner {
    let name = "vecLib cblas_sgemm"

    func reference(a: Matrix, b: Matrix, problem: GEMMProblem) -> Matrix {
        var output = [Float](repeating: 0, count: problem.outputElementCount)
        runSGEMM(a: a.elements, b: b.elements, c: &output, problem: problem)
        return Matrix(rows: problem.m, cols: problem.n, elements: output)
    }

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        var output = [Float](repeating: 0, count: problem.outputElementCount)
        var timings = [Double]()

        for iteration in 0..<(warmupIterations + measuredIterations) {
            let (_, elapsedMs) = measureMilliseconds {
                runSGEMM(a: a.elements, b: b.elements, c: &output, problem: problem)
            }
            if iteration >= warmupIterations {
                timings.append(elapsedMs)
            }
        }

        return RawBenchmarkRun(
            output: Matrix(rows: problem.m, cols: problem.n, elements: output),
            wallAverageMs: timings.average,
            wallBestMs: timings.min() ?? 0,
            deviceAverageMs: nil,
            deviceBestMs: nil
        )
    }

    private func runSGEMM(a: [Float], b: [Float], c: inout [Float], problem: GEMMProblem) {
        a.withUnsafeBufferPointer { aPointer in
            b.withUnsafeBufferPointer { bPointer in
                c.withUnsafeMutableBufferPointer { cPointer in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        Int32(problem.m),
                        Int32(problem.n),
                        Int32(problem.k),
                        1.0,
                        aPointer.baseAddress!,
                        Int32(problem.k),
                        bPointer.baseAddress!,
                        Int32(problem.n),
                        0.0,
                        cPointer.baseAddress!,
                        Int32(problem.n)
                    )
                }
            }
        }
    }
}
