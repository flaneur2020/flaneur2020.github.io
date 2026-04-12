import Foundation

final class MetalBestGEMMRunner: GEMMRunner {
    private struct Candidate {
        let name: String
        let runner: MetalTiledGEMMRunner
    }

    let name = "Metal best (autotuned)"

    private let candidates: [Candidate]
    private let sampleWarmupIterations: Int
    private let sampleMeasuredIterations: Int
    private var selectedCandidateIndexes = [GEMMProblem: Int]()

    init(
        candidates: [(name: String, runner: MetalTiledGEMMRunner)],
        sampleWarmupIterations: Int = 1,
        sampleMeasuredIterations: Int = 5
    ) {
        precondition(!candidates.isEmpty)
        self.candidates = candidates.map { candidate in
            Candidate(name: candidate.name, runner: candidate.runner)
        }
        self.sampleWarmupIterations = sampleWarmupIterations
        self.sampleMeasuredIterations = sampleMeasuredIterations
    }

    func supports(problem: GEMMProblem) -> Bool {
        candidates.contains { $0.runner.supports(problem: problem) }
    }

    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun {
        let candidate = try selectCandidate(a: a, b: b, problem: problem)
        return try candidate.runner.benchmark(
            a: a,
            b: b,
            problem: problem,
            warmupIterations: warmupIterations,
            measuredIterations: measuredIterations
        )
    }

    func variantDescription(for problem: GEMMProblem) -> String? {
        guard let index = selectedCandidateIndexes[problem] else {
            return nil
        }
        return candidates[index].name
    }

    private func selectCandidate(a: Matrix, b: Matrix, problem: GEMMProblem) throws -> Candidate {
        if let cachedIndex = selectedCandidateIndexes[problem] {
            return candidates[cachedIndex]
        }

        let preferredCandidates = preferredCandidateIndexes(for: problem)

        var bestIndex: Int?
        var bestScoreMs = Double.greatestFiniteMagnitude

        for index in preferredCandidates {
            let candidate = candidates[index]
            guard candidate.runner.supports(problem: problem) else {
                continue
            }

            let sampleRun = try candidate.runner.benchmark(
                a: a,
                b: b,
                problem: problem,
                warmupIterations: sampleWarmupIterations,
                measuredIterations: sampleMeasuredIterations
            )
            let scoreMs = sampleRun.deviceAverageMs ?? sampleRun.wallAverageMs

            if scoreMs < bestScoreMs {
                bestScoreMs = scoreMs
                bestIndex = index
            }
        }

        guard let bestIndex else {
            throw BenchmarkError.runtimeFailure("No Metal-best candidate supports \(problem.description)")
        }

        selectedCandidateIndexes[problem] = bestIndex
        return candidates[bestIndex]
    }

    private func preferredCandidateIndexes(for problem: GEMMProblem) -> [Int] {
        let aligned64x64 = problem.m % 64 == 0 && problem.n % 64 == 0 && problem.k % 16 == 0
        let aligned64x64k32 = problem.m % 64 == 0 && problem.n % 64 == 0 && problem.k % 32 == 0
        let aligned64x32 = problem.m % 64 == 0 && problem.n % 32 == 0 && problem.k % 16 == 0
        let aligned32x32 = problem.m % 32 == 0 && problem.n % 32 == 0 && problem.k % 16 == 0

        let preferredNames: [String]
        if aligned64x64k32 {
            preferredNames = ["64x64x16", "64x64x32"]
        } else if aligned64x64 {
            preferredNames = ["64x64x16"]
        } else if aligned64x32 {
            preferredNames = ["64x32"]
        } else if aligned32x32 {
            preferredNames = ["4x4 aligned", "4x4 k16"]
        } else {
            preferredNames = ["4x4 k16"]
        }

        let preferred = candidates.enumerated().compactMap { index, candidate -> Int? in
            guard preferredNames.contains(where: candidate.name.contains) else {
                return nil
            }
            return candidate.runner.supports(problem: problem) ? index : nil
        }

        if !preferred.isEmpty {
            return preferred
        }

        return candidates.enumerated().compactMap { index, candidate in
            candidate.runner.supports(problem: problem) ? index : nil
        }
    }
}
