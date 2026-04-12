import Foundation

let vecLibRunner = VecLibGEMMRunner()
let metalRunnerConfigurations = [
    (name: "Metal tiled 16x16", functionName: "tiled_gemm_16x16", tileSize: 16),
    (name: "Metal tiled 32x32", functionName: "tiled_gemm_32x32", tileSize: 32),
    (name: "Metal swizzled 32x32", functionName: "swizzled_gemm_32x32", tileSize: 32),
]

do {
    guard let configuration = try CLI.parse(arguments: CommandLine.arguments) else {
        print(CLI.usage)
        exit(0)
    }

    var runners: [any GEMMRunner] = [vecLibRunner]
    for runnerConfiguration in metalRunnerConfigurations {
        do {
            runners.append(
                try MetalTiledGEMMRunner(
                    name: runnerConfiguration.name,
                    functionName: runnerConfiguration.functionName,
                    tileSize: runnerConfiguration.tileSize
                )
            )
        } catch {
            fputs("warning: skipping \(runnerConfiguration.name): \(error.localizedDescription)\n", stderr)
        }
    }

    print("Benchmarking GEMM with X = MNK and Y = MFLOPs")
    print("Baseline: Apple vecLib via Accelerate cblas_sgemm")
    print("")
    print(
        "\(pad("implementation", to: 22)) \(pad("problem", to: 14)) \(pad("MNK", to: 14)) \(pad("avg ms", to: 12)) \(pad("best ms", to: 12)) \(pad("MFLOPs", to: 14)) max |Δ|"
    )

    var measurements = [BenchmarkMeasurement]()

    for problem in configuration.problems.sorted() {
        var generator = SplitMix64(seed: seed(for: problem))
        let a = Matrix.random(rows: problem.m, cols: problem.k, generator: &generator)
        let b = Matrix.random(rows: problem.k, cols: problem.n, generator: &generator)
        let reference = vecLibRunner.reference(a: a, b: b, problem: problem)

        for runner in runners {
            let run = try runner.benchmark(
                a: a,
                b: b,
                problem: problem,
                warmupIterations: configuration.warmupIterations,
                measuredIterations: configuration.measuredIterations
            )

            let measurement = BenchmarkMeasurement(
                implementation: runner.name,
                problem: problem,
                averageMs: run.averageMs,
                bestMs: run.bestMs,
                mflops: problem.mflops(forMilliseconds: run.averageMs),
                maxAbsError: run.output.maxAbsoluteDifference(comparedTo: reference)
            )
            measurements.append(measurement)

            print(
                "\(pad(measurement.implementation, to: 22)) " +
                "\(pad(problem.description, to: 14)) " +
                "\(pad(String(problem.mnkProduct), to: 14)) " +
                "\(pad(formatMilliseconds(measurement.averageMs), to: 12)) " +
                "\(pad(formatMilliseconds(measurement.bestMs), to: 12)) " +
                "\(pad(formatMFLOPs(measurement.mflops), to: 14)) " +
                "\(formatError(measurement.maxAbsError))"
            )
        }
    }

    if let csvPath = configuration.csvPath {
        try writeCSV(measurements, to: csvPath)
        print("\nWrote CSV data to \(csvPath)")
    }
} catch {
    fputs("error: \(error.localizedDescription)\n", stderr)
    exit(1)
}

private func seed(for problem: GEMMProblem) -> UInt64 {
    let upperM = UInt64(problem.m) << 42
    let upperN = UInt64(problem.n) << 21
    let lowerK = UInt64(problem.k)
    return upperM ^ upperN ^ lowerK ^ 0x9E3779B97F4A7C15
}

private func writeCSV(_ measurements: [BenchmarkMeasurement], to path: String) throws {
    var lines = [
        "implementation,m,n,k,mnk,average_ms,best_ms,mflops,max_abs_error",
    ]

    for measurement in measurements {
        lines.append(
            [
                measurement.implementation,
                String(measurement.problem.m),
                String(measurement.problem.n),
                String(measurement.problem.k),
                String(measurement.problem.mnkProduct),
                String(format: "%.6f", measurement.averageMs),
                String(format: "%.6f", measurement.bestMs),
                String(format: "%.6f", measurement.mflops),
                String(format: "%.8e", Double(measurement.maxAbsError)),
            ].joined(separator: ",")
        )
    }

    try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
}
