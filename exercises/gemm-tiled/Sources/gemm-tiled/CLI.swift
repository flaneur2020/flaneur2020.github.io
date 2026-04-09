import Foundation

enum CLI {
    static let usage = """
    Usage:
      swift run gemm-tiled [options]

    Options:
      --problems LIST      Comma-separated GEMM sizes, like 128 or 256x256x256
      --iterations N       Measured iterations per problem (default: 10)
      --warmup N           Warmup iterations per problem (default: 2)
      --csv PATH           Write CSV output to PATH (default: benchmark.csv)
      --no-csv             Disable CSV output
      --help               Show this help message

    Notes:
      - The benchmark reports X-axis data as MNK = M*N*K.
      - The benchmark reports Y-axis data as MFLOPs.
      - The baseline uses Apple vecLib through Accelerate cblas_sgemm.
    """

    static func parse(arguments: [String]) throws -> BenchmarkConfiguration? {
        var problems = [128, 256, 384, 512, 768, 1024].map {
            GEMMProblem(m: $0, n: $0, k: $0)
        }
        var measuredIterations = 10
        var warmupIterations = 2
        var csvPath: String? = "benchmark.csv"

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--help", "-h":
                return nil
            case "--problems":
                index += 1
                guard index < arguments.count else {
                    throw BenchmarkError.invalidArgument("Missing value for --problems")
                }
                problems = try parseProblems(arguments[index])
            case "--iterations":
                index += 1
                guard index < arguments.count else {
                    throw BenchmarkError.invalidArgument("Missing value for --iterations")
                }
                measuredIterations = try parsePositiveInt(arguments[index], flag: "--iterations")
            case "--warmup":
                index += 1
                guard index < arguments.count else {
                    throw BenchmarkError.invalidArgument("Missing value for --warmup")
                }
                warmupIterations = try parsePositiveInt(arguments[index], flag: "--warmup")
            case "--csv":
                index += 1
                guard index < arguments.count else {
                    throw BenchmarkError.invalidArgument("Missing value for --csv")
                }
                csvPath = arguments[index]
            case "--no-csv":
                csvPath = nil
            default:
                throw BenchmarkError.invalidArgument("Unknown argument: \(argument)")
            }
            index += 1
        }

        return BenchmarkConfiguration(
            problems: problems.sorted(),
            warmupIterations: warmupIterations,
            measuredIterations: measuredIterations,
            csvPath: csvPath
        )
    }

    private static func parseProblems(_ value: String) throws -> [GEMMProblem] {
        let items = value.split(separator: ",").map(String.init)
        guard !items.isEmpty else {
            throw BenchmarkError.invalidArgument("--problems cannot be empty")
        }
        return try items.map(parseProblem)
    }

    private static func parseProblem(_ value: String) throws -> GEMMProblem {
        let dimensions = value.lowercased().split(separator: "x")

        if dimensions.count == 1 {
            let size = try parsePositiveInt(String(dimensions[0]), flag: "--problems")
            return GEMMProblem(m: size, n: size, k: size)
        }

        guard dimensions.count == 3 else {
            throw BenchmarkError.invalidArgument("Problem sizes must look like 256 or 256x256x256")
        }

        let m = try parsePositiveInt(String(dimensions[0]), flag: "--problems")
        let n = try parsePositiveInt(String(dimensions[1]), flag: "--problems")
        let k = try parsePositiveInt(String(dimensions[2]), flag: "--problems")
        return GEMMProblem(m: m, n: n, k: k)
    }

    private static func parsePositiveInt(_ value: String, flag: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw BenchmarkError.invalidArgument("\(flag) expects a positive integer, got \(value)")
        }
        return parsed
    }
}
