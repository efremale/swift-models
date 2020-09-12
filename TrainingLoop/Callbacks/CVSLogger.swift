import Foundation
import ModelSupport

public class CVSLogger {
	public var path: String
	public var liveStatistics: Bool
    
    var foundationFS: FoundationFileSystem
	var foundationFile: FoundationFile

	public init(withPath path: String = "run/log.csv", liveStatistics: Bool = true) {
		self.path = path
		self.liveStatistics = liveStatistics
		self.foundationFS = FoundationFileSystem()
		self.foundationFile = FoundationFile(path: path)
	}

	public func log<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
		switch event {
		case .batchEnd:
			guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount, 
			let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else {
				break
			}

			if !liveStatistics && (batchIndex + 1 != batchCount) {
				break
			}

			guard let stats = loop.lastStatsLog else {
				break
			}

			if !FileManager.default.fileExists(atPath: path) {
				try foundationFS.createDirectoryIfMissing(at: String(path[..<path.lastIndex(of: "/")!]))
				try writeHeader(stats: stats)
			}
			try writeDataRow(
				epoch: "\(epochIndex + 1)/\(epochCount)",
				batch: "\(batchIndex + 1)/\(batchCount)",
				stats: stats)
		default: 
			break	
		}
	}

	func writeHeader(stats: [(String, Float)]) throws {
		let head: String = (["epoch", "batch"] + stats.map { $0.0 }).joined(separator: ", ")
		do {
			try head.write(toFile: path, atomically: true, encoding: .utf8)
		} catch {
			print("Unexpected error in writing header line: \(error).")
			throw error
		}
	}

	func writeDataRow(epoch: String, batch: String, stats: [(String, Float)]) throws {
		let dataRow: Data = ("\n" + ([epoch, batch] + stats.map { String($0.1) }).joined(separator: ", ")).data(using: .utf8)!
		do {
			try foundationFile.append(dataRow)
		} catch {
			print("Unexpected error in writing data row: \(error).")
			throw error
		}
	}
}