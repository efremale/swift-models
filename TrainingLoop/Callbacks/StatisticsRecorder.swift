// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import TensorFlow


/// A callback-based handler of statistics obtained during a training loop. This can be employed
/// by progress bars, recorders, or logging functionality.
public class StatisticsRecorder {
  public var liveStatistics: Bool

  var metricMeasurers: [MetricsMeasurer]

  /// Initializes the statistics tracker with
  ///
  /// - Parameters:
  ///   - metrics: A set of TrainingMetrics to capture during the training loop.
  public init(liveStatistics: Bool = true, metrics: [TrainingMetrics]) {
    self.liveStatistics = liveStatistics
    metricMeasurers = [(TrainingMetrics.loss).measurer] + metrics.map { $0.measurer }
  }

  /// The callback used to hook into the TrainingLoop. This is updated once per event.
  ///
  /// - Parameters:
  ///   - loop: The TrainingLoop where an event has occurred. This can be accessed to obtain
  ///     the last measure loss and other values.
  ///   - event: The training or validation event that this callback is responding to.
  public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .trainingStart, .validationStart:
      resetMetricMeasurers()
    case .batchEnd:
      if let loss = loop.lastStepLoss, let output = loop.lastStepOutput, let target = loop.lastStepTarget {
        accumulateMetrics(loss: loss, predictions: output, labels: target)
      }
      
      if let batchIndex = loop.batchIndex, let batchCount = loop.batchCount {
        if liveStatistics || (batchCount == (batchIndex + 1)) {
          loop.lastStatsLog = computeMetrics()
        }
      }
    default:
      return
    }
  }

  func resetMetricMeasurers() {
    for index in metricMeasurers.indices {
      metricMeasurers[index].reset()
    }
  }

  func accumulateMetrics<Output, Target>(loss: Tensor<Float>, predictions: Output, labels: Target) {
    for index in metricMeasurers.indices {
      metricMeasurers[index].accumulate(loss: loss, predictions: predictions, labels: labels) 
    }
  }

  func computeMetrics() -> [(String, Float)] {
    var result: [(String, Float)] = []
    for measurer in metricMeasurers {
      result.append((measurer.name, measurer.measure()))
    }
    return result
  }
}
