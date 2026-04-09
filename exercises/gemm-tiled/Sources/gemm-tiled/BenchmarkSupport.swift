import Dispatch
import Foundation

struct GEMMProblem: Comparable, CustomStringConvertible {
    let m: Int
    let n: Int
    let k: Int

    var description: String {
        "\(m)x\(n)x\(k)"
    }

    var mnkProduct: Int {
        m * n * k
    }

    var outputElementCount: Int {
        m * n
    }

    func mflops(forMilliseconds milliseconds: Double) -> Double {
        guard milliseconds > 0 else {
            return 0
        }

        let totalFloatingPointOperations = 2.0 * Double(mnkProduct)
        return totalFloatingPointOperations / (milliseconds / 1_000.0) / 1_000_000.0
    }

    static func < (lhs: GEMMProblem, rhs: GEMMProblem) -> Bool {
        lhs.mnkProduct < rhs.mnkProduct
    }
}

struct Matrix {
    let rows: Int
    let cols: Int
    let elements: [Float]

    init(rows: Int, cols: Int, elements: [Float]) {
        precondition(elements.count == rows * cols)
        self.rows = rows
        self.cols = cols
        self.elements = elements
    }

    static func random(rows: Int, cols: Int, generator: inout SplitMix64) -> Matrix {
        let count = rows * cols
        let elements = (0..<count).map { _ in
            generator.nextFloat(in: -1.0 ... 1.0)
        }
        return Matrix(rows: rows, cols: cols, elements: elements)
    }

    func maxAbsoluteDifference(comparedTo other: Matrix) -> Float {
        precondition(rows == other.rows && cols == other.cols)

        var difference: Float = 0
        for index in elements.indices {
            difference = max(difference, abs(elements[index] - other.elements[index]))
        }
        return difference
    }
}

struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = Double(next()) / Double(UInt64.max)
        return range.lowerBound + Float(unit) * (range.upperBound - range.lowerBound)
    }
}

struct BenchmarkConfiguration {
    let problems: [GEMMProblem]
    let warmupIterations: Int
    let measuredIterations: Int
    let csvPath: String?
}

struct RawBenchmarkRun {
    let output: Matrix
    let averageMs: Double
    let bestMs: Double
}

struct BenchmarkMeasurement {
    let implementation: String
    let problem: GEMMProblem
    let averageMs: Double
    let bestMs: Double
    let mflops: Double
    let maxAbsError: Float
}

protocol GEMMRunner {
    var name: String { get }
    func benchmark(
        a: Matrix,
        b: Matrix,
        problem: GEMMProblem,
        warmupIterations: Int,
        measuredIterations: Int
    ) throws -> RawBenchmarkRun
}

enum BenchmarkError: LocalizedError {
    case invalidArgument(String)
    case unsupportedPlatform(String)
    case runtimeFailure(String)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return message
        case .unsupportedPlatform(let message):
            return message
        case .runtimeFailure(let message):
            return message
        }
    }
}

func measureMilliseconds<T>(_ work: () throws -> T) rethrows -> (T, Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let value = try work()
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
    return (value, elapsed)
}

extension Array where Element == Double {
    var average: Double {
        guard !isEmpty else {
            return 0
        }
        return reduce(0, +) / Double(count)
    }
}

func pad(_ value: String, to width: Int) -> String {
    guard value.count < width else {
        return value
    }
    return value + String(repeating: " ", count: width - value.count)
}

func formatMilliseconds(_ value: Double) -> String {
    String(format: "%.3f", value)
}

func formatMFLOPs(_ value: Double) -> String {
    String(format: "%.1f", value)
}

func formatError(_ value: Float) -> String {
    String(format: "%.2e", Double(value))
}
