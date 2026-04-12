import Foundation

let vecLibRunner = VecLibGEMMRunner()
let metalRunnerConfigurations = [
    MetalKernelConfiguration(
        name: "Metal tiled 16x16",
        functionName: "tiled_gemm_16x16",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 16,
        outputTileHeight: 16,
        aOperandLayout: .rowMajor,
        bOperandLayout: .rowMajor
    ),
    MetalKernelConfiguration(
        name: "Metal tiled 32x32",
        functionName: "tiled_gemm_32x32",
        threadgroupWidth: 32,
        threadgroupHeight: 32,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .rowMajor
    ),
    MetalKernelConfiguration(
        name: "Metal swizzled 32x32",
        functionName: "swizzled_gemm_32x32",
        threadgroupWidth: 32,
        threadgroupHeight: 32,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .rowMajor
    ),
    MetalKernelConfiguration(
        name: "Metal register blocked 4x4",
        functionName: "register_blocked_gemm_4x4",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .rowMajor
    ),
    MetalKernelConfiguration(
        name: "Metal packed-swizzled B 4x4",
        functionName: "packed_swizzled_b_gemm_4x4",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .packedSwizzled(blockK: 8, blockN: 32, swizzleGroup: 8)
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized B 4x4",
        functionName: "packed_vectorized_b_gemm_4x4",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .packedVectorized(blockK: 8, blockN: 32, vectorWidth: 4)
    ),
    MetalKernelConfiguration(
        name: "Metal packed-swizzled vec4 B 4x4",
        functionName: "packed_swizzled_vectorized_b_gemm_4x4",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .packedVectorizedSwizzled(blockK: 8, blockN: 32, vectorWidth: 4, vectorSwizzleGroup: 4)
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized B 4x4 k16",
        functionName: "packed_vectorized_b_gemm_4x4_k16",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .rowMajor,
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4)
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 4x4 k16",
        functionName: "packed_vectorized_a_b_gemm_4x4_k16",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4)
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 4x4 aligned",
        functionName: "packed_vectorized_a_b_gemm_4x4_k16_aligned",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 4x4 aligned private",
        functionName: "packed_vectorized_a_b_gemm_4x4_k16_aligned",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        bufferMode: .privateStaged,
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x32x16",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x32x16_aligned",
        threadgroupWidth: 8,
        threadgroupHeight: 16,
        outputTileWidth: 32,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x64x16",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x64x16_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 64, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x64x16 private",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x64x16_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 64, vectorWidth: 4),
        bufferMode: .privateStaged,
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x64x16 unroll",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x64x16_unrolled",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 64, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 8x4 64x64x16",
        functionName: "packed_vectorized_a_b_gemm_8x4_64x64x16_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 8,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 64, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 8x4 64x128x16",
        functionName: "packed_vectorized_a_b_gemm_8x4_64x128x16_aligned",
        threadgroupWidth: 32,
        threadgroupHeight: 8,
        outputTileWidth: 128,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 128, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x64x32",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x64x32_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 32, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 32, blockN: 64, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 32
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x64x32 private",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x64x32_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 16,
        outputTileWidth: 64,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 32, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 32, blockN: 64, vectorWidth: 4),
        bufferMode: .privateStaged,
        requiresAlignedProblem: true,
        requiredKAlignment: 32
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 64x32x16 unroll",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x32x16_unrolled",
        threadgroupWidth: 8,
        threadgroupHeight: 16,
        outputTileWidth: 32,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed A+B 64x32x16 pipe",
        functionName: "packed_vectorized_a_b_gemm_4x4_64x32x16_pipelined",
        threadgroupWidth: 8,
        threadgroupHeight: 16,
        outputTileWidth: 32,
        outputTileHeight: 64,
        aOperandLayout: .packedVectorized(blockM: 64, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 32x64x16",
        functionName: "packed_vectorized_a_b_gemm_4x4_32x64x16_aligned",
        threadgroupWidth: 16,
        threadgroupHeight: 8,
        outputTileWidth: 64,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 64, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 32x32x32",
        functionName: "packed_vectorized_a_b_gemm_4x4_32x32x32_aligned",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 32, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 32, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 32
    ),
    MetalKernelConfiguration(
        name: "Metal packed A+B aligned pipe",
        functionName: "packed_vectorized_a_b_gemm_4x4_k16_aligned_pipelined",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4),
        requiresAlignedProblem: true,
        requiredKAlignment: 16
    ),
    MetalKernelConfiguration(
        name: "Metal packed-vectorized A+B 4x4 unroll",
        functionName: "packed_vectorized_a_b_gemm_4x4_k16_unrolled",
        threadgroupWidth: 8,
        threadgroupHeight: 8,
        outputTileWidth: 32,
        outputTileHeight: 32,
        aOperandLayout: .packedVectorized(blockM: 32, blockK: 16, vectorHeight: 4),
        bOperandLayout: .packedVectorized(blockK: 16, blockN: 32, vectorWidth: 4)
    ),
]

do {
    guard let configuration = try CLI.parse(arguments: CommandLine.arguments) else {
        print(CLI.usage)
        exit(0)
    }

    let runners = try buildRunners(matching: configuration.implementationFilters)

    print("Benchmarking GEMM with X = MNK and Y = MFLOPs")
    print("Primary metric: unified wall time across vecLib and Metal")
    print("Metal GPU timestamps are exported to CSV as secondary columns when available")
    print("Baseline: Apple vecLib via Accelerate cblas_sgemm")
    print("")
    print(
        "\(pad("implementation", to: 40)) \(pad("problem", to: 14)) \(pad("MNK", to: 14)) \(pad("wall avg", to: 12)) \(pad("wall best", to: 12)) \(pad("wall MFLOPs", to: 14)) max |Δ|"
    )

    var measurements = [BenchmarkMeasurement]()

    for (problemIndex, problem) in configuration.problems.sorted().enumerated() {
        var generator = SplitMix64(seed: seed(for: problem))
        let a = Matrix.random(rows: problem.m, cols: problem.k, generator: &generator)
        let b = Matrix.random(rows: problem.k, cols: problem.n, generator: &generator)
        let reference = vecLibRunner.reference(a: a, b: b, problem: problem)

        let orderedRunners = problemIndex.isMultiple(of: 2) ? runners : Array(runners.reversed())

        for runner in orderedRunners {
            guard runner.supports(problem: problem) else {
                fputs("warning: skipping \(runner.name) on \(problem.description): requires aligned problem dimensions\n", stderr)
                continue
            }

            let run = try runner.benchmark(
                a: a,
                b: b,
                problem: problem,
                warmupIterations: configuration.warmupIterations,
                measuredIterations: configuration.measuredIterations
            )

            let measurement = BenchmarkMeasurement(
                implementation: runner.name,
                selectedVariant: runner.variantDescription(for: problem),
                problem: problem,
                wallAverageMs: run.wallAverageMs,
                wallBestMs: run.wallBestMs,
                wallMflops: problem.mflops(forMilliseconds: run.wallAverageMs),
                deviceAverageMs: run.deviceAverageMs,
                deviceBestMs: run.deviceBestMs,
                deviceMflops: run.deviceAverageMs.map { problem.mflops(forMilliseconds: $0) },
                maxAbsError: run.output.maxAbsoluteDifference(comparedTo: reference)
            )
            measurements.append(measurement)

            print(
                "\(pad(measurement.implementation, to: 40)) " +
                "\(pad(problem.description, to: 14)) " +
                "\(pad(String(problem.mnkProduct), to: 14)) " +
                "\(pad(formatMilliseconds(measurement.wallAverageMs), to: 12)) " +
                "\(pad(formatMilliseconds(measurement.wallBestMs), to: 12)) " +
                "\(pad(formatMFLOPs(measurement.wallMflops), to: 14)) " +
                "\(formatError(measurement.maxAbsError))"
            )
        }
    }

    if let csvPath = configuration.csvPath {
        try writeCSV(measurements, to: csvPath)
        print("\nWrote CSV data to \(csvPath)")
    }

    printComparisonSummary(measurements)
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
        "implementation,selected_variant,m,n,k,mnk,average_ms,best_ms,mflops,device_average_ms,device_best_ms,device_mflops,max_abs_error",
    ]

    for measurement in measurements {
        lines.append(
            [
                measurement.implementation,
                measurement.selectedVariant ?? "",
                String(measurement.problem.m),
                String(measurement.problem.n),
                String(measurement.problem.k),
                String(measurement.problem.mnkProduct),
                String(format: "%.6f", measurement.wallAverageMs),
                String(format: "%.6f", measurement.wallBestMs),
                String(format: "%.6f", measurement.wallMflops),
                csvValue(measurement.deviceAverageMs),
                csvValue(measurement.deviceBestMs),
                csvValue(measurement.deviceMflops),
                String(format: "%.8e", Double(measurement.maxAbsError)),
            ].joined(separator: ",")
        )
    }

    try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
}


private func buildRunners(matching filters: [String]?) throws -> [any GEMMRunner] {
    let normalizedFilters = normalizeImplementationFilters(filters)

    if normalizedFilters == nil || normalizedFilters?.contains("all") == true {
        return try buildDefaultRunners()
    }

    var runners = [any GEMMRunner]()

    if shouldIncludeVecLib(in: normalizedFilters) {
        runners.append(vecLibRunner)
    }

    if shouldIncludeMPS(in: normalizedFilters) {
        do {
            runners.append(try MPSGEMMRunner())
        } catch {
            fputs("warning: skipping MPSMatrixMultiplication: \(error.localizedDescription)\n", stderr)
        }
    }

    if shouldIncludeMetalBest(in: normalizedFilters) {
        if let bestRunner = try makeMetalBestRunner() {
            runners.append(bestRunner)
        }
    }

    if shouldIncludeMPP(in: normalizedFilters) {
        if #available(macOS 26.0, *) {
            do {
                runners.append(try MetalMPPGEMMRunner())
            } catch {
                fputs("warning: skipping Metal MPP single-tile 64x32 (k<64): \(error.localizedDescription)\n", stderr)
            }
        }
    }

    for runnerConfiguration in metalRunnerConfigurations where shouldInclude(runnerConfiguration, in: normalizedFilters) {
        do {
            runners.append(try MetalTiledGEMMRunner(configuration: runnerConfiguration))
        } catch {
            fputs("warning: skipping \(runnerConfiguration.name): \(error.localizedDescription)\n", stderr)
        }
    }

    reportUnmatchedImplementationFilters(normalizedFilters)

    guard !runners.isEmpty else {
        throw BenchmarkError.invalidArgument("No implementations matched --implementations")
    }

    return runners
}

private func buildDefaultRunners() throws -> [any GEMMRunner] {
    var runners: [any GEMMRunner] = [vecLibRunner]

    do {
        runners.append(try MPSGEMMRunner())
    } catch {
        fputs("warning: skipping MPSMatrixMultiplication: \(error.localizedDescription)\n", stderr)
    }

    if #available(macOS 26.0, *) {
        do {
            runners.append(try MetalMPPGEMMRunner())
        } catch {
            fputs("warning: skipping Metal MPP single-tile 64x32 (k<64): \(error.localizedDescription)\n", stderr)
        }
    }

    for runnerConfiguration in metalRunnerConfigurations {
        do {
            runners.append(try MetalTiledGEMMRunner(configuration: runnerConfiguration))
        } catch {
            fputs("warning: skipping \(runnerConfiguration.name): \(error.localizedDescription)\n", stderr)
        }
    }

    return runners
}

private func makeMetalBestRunner() throws -> MetalBestGEMMRunner? {
    let candidateNames = [
        "Metal packed-vectorized A+B 64x64x16",
        "Metal packed-vectorized A+B 64x64x16 private",
        "Metal packed-vectorized A+B 64x64x32",
        "Metal packed-vectorized A+B 64x64x32 private",
        "Metal packed-vectorized A+B 64x32x16",
        "Metal packed-vectorized A+B 4x4 aligned",
        "Metal packed-vectorized A+B 4x4 k16",
    ]

    var candidates = [(name: String, runner: MetalTiledGEMMRunner)]()

    for candidateName in candidateNames {
        guard let runnerConfiguration = metalRunnerConfigurations.first(where: { $0.name == candidateName }) else {
            continue
        }

        do {
            candidates.append((candidateName, try MetalTiledGEMMRunner(configuration: runnerConfiguration)))
        } catch {
            fputs("warning: skipping \(candidateName) for metal-best: \(error.localizedDescription)\n", stderr)
        }
    }

    guard !candidates.isEmpty else {
        return nil
    }

    return MetalBestGEMMRunner(candidates: candidates)
}

private func normalizeImplementationFilters(_ filters: [String]?) -> [String]? {
    guard let filters else {
        return nil
    }

    let normalized = filters
        .map(normalizedImplementationKey)
        .filter { !$0.isEmpty }

    return normalized.isEmpty ? nil : normalized
}

private func shouldIncludeVecLib(in filters: [String]?) -> Bool {
    guard let filters else {
        return true
    }

    return filters.contains(where: isVecLibFilter)
}

private func shouldIncludeMPS(in filters: [String]?) -> Bool {
    guard let filters else {
        return false
    }

    return filters.contains(where: isMPSFilter)
}

private func shouldIncludeMetalBest(in filters: [String]?) -> Bool {
    guard let filters else {
        return false
    }

    return filters.contains(where: isMetalBestFilter)
}

private func shouldIncludeMPP(in filters: [String]?) -> Bool {
    guard let filters else {
        return false
    }

    return filters.contains(where: isMPPFilter)
}

private func shouldInclude(_ configuration: MetalKernelConfiguration, in filters: [String]?) -> Bool {
    guard let filters else {
        return true
    }

    let normalizedName = normalizedImplementationKey(configuration.name)
    return filters.contains { filter in
        isAllMetalFilter(filter) || normalizedName.contains(filter)
    }
}

private func reportUnmatchedImplementationFilters(_ filters: [String]?) {
    guard let filters else {
        return
    }

    let unmatched = filters.filter { filter in
        !matchesAnyImplementation(filter)
    }

    guard !unmatched.isEmpty else {
        return
    }

    fputs("warning: no implementations matched filter(s): \(unmatched.joined(separator: ", "))\n", stderr)
}

private func matchesAnyImplementation(_ filter: String) -> Bool {
    if filter == "all" || isVecLibFilter(filter) || isMPSFilter(filter) || isMetalBestFilter(filter) || isMPPFilter(filter) || isAllMetalFilter(filter) {
        return true
    }

    return metalRunnerConfigurations.contains { normalizedImplementationKey($0.name).contains(filter) }
}

private func isVecLibFilter(_ filter: String) -> Bool {
    ["veclib", "vec lib", "baseline", "cblas", "sgemm", "accelerate"].contains(filter)
}

private func isMPSFilter(_ filter: String) -> Bool {
    ["mps", "mpsmatrix", "mpsmatrixmultiplication", "matrixmultiplication", "metal performance shaders"].contains(filter)
}

private func isMetalBestFilter(_ filter: String) -> Bool {
    ["metal best", "best metal", "best", "autotuned", "autotune"].contains(filter)
}

private func isMPPFilter(_ filter: String) -> Bool {
    ["mpp", "cooperative", "cooperative tensor", "tensor ops", "tensorops"].contains(filter)
}

private func isAllMetalFilter(_ filter: String) -> Bool {
    ["metal", "metal all"].contains(filter)
}

private func normalizedImplementationKey(_ value: String) -> String {
    let lowercased = value.lowercased()
    let normalizedCharacters = lowercased.map { character -> Character in
        character.isLetter || character.isNumber ? character : " "
    }
    return String(normalizedCharacters)
        .split(whereSeparator: \.isWhitespace)
        .joined(separator: " ")
}

private func printComparisonSummary(_ measurements: [BenchmarkMeasurement]) {
    let groupedByProblem = Dictionary(grouping: measurements, by: \.problem)
    let baselineName = vecLibRunner.name

    let summaryRows = groupedByProblem.keys.sorted().compactMap { problem -> String? in
        guard let baseline = groupedByProblem[problem]?.first(where: { $0.implementation == baselineName }) else {
            return nil
        }

        let competitors = groupedByProblem[problem]?
            .filter { $0.implementation != baselineName }
            .sorted { $0.wallMflops > $1.wallMflops } ?? []

        guard let bestCompetitor = competitors.first else {
            return nil
        }

        let speedup = bestCompetitor.wallMflops / baseline.wallMflops - 1.0
        let variantSuffix: String
        if let selectedVariant = bestCompetitor.selectedVariant, selectedVariant != bestCompetitor.implementation {
            variantSuffix = " via \(selectedVariant)"
        } else {
            variantSuffix = ""
        }

        return "\(pad(problem.description, to: 14)) \(pad(bestCompetitor.implementation, to: 28)) \(formatSpeedup(speedup))\(variantSuffix)"
    }

    guard !summaryRows.isEmpty else {
        return
    }

    print("\nBest-vs-vecLib summary")
    for row in summaryRows {
        print(row)
    }
}

private func formatSpeedup(_ value: Double) -> String {
    String(format: "%+.2f%%", value * 100.0)
}
