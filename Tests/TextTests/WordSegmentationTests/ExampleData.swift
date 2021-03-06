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

import ModelSupport
import TensorFlow
import TextModels

// Generated by `blaze run //experimental/users/marcrasi/probe_wordseg:probe`
enum Example1 {
  static let parameters = SNLMParameters(
    emb_enc: EmbeddingParameters(
      weight: Tensor<Float>(
        [
          [-0.03872304, -0.05338321],
          [0.051871493, -0.024247635],
          [-0.05366139, 0.05159463],
          [0.047521368, 0.010823324],
          [-0.0012689009, -0.07714497],
        ]
      )
    ),
    lstm_enc: LSTMParameters(
      weight_ih_l0: Tensor<Float>(
        [
          [0.029065304, 0.035071626],
          [0.05220075, 0.07659577],
          [-0.03418843, 0.054938704],
          [-0.076050006, 0.011613108],
          [-0.07208062, -0.028602105],
          [0.07454459, -0.045288637],
          [0.056929305, 0.07525672],
          [-0.044981938, -0.06287278],
        ]
      ),
      weight_hh_l0: Tensor<Float>(
        [
          [0.054794624, -0.047684688],
          [0.059219867, 0.035521813],
          [0.07620007, -0.024476558],
          [0.06970246, -0.012173824],
          [-0.01848751, -0.040551327],
          [0.050961852, 0.03887064],
          [0.054652542, -0.015333958],
          [0.048499897, 0.044002943],
        ]
      ),
      bias_ih_l0: Tensor<Float>(
        [
          -0.061996475, 0.07747544, 0.04455457, -0.016387768, -0.07249636,
          0.047923, -0.040993594, -0.011194035,
        ]
      ),
      bias_hh_l0: Tensor<Float>(
        [
          1.9561797e-02, -4.7253761e-02, -1.0812007e-02, 7.7792764e-02,
          -3.5792589e-05, -3.5584867e-02, 5.0887465e-05, -3.5637055e-02,
        ]
      )
    ),
    mlp_interpolation: MLPParameters(
      linear1: LinearParameters(
        weight: Tensor<Float>(
          [
            [-0.034370955, -0.07964967],
            [0.019697152, -0.059639234],
          ]
        ),
        bias: Tensor<Float>(
          [0.047385678, 0.05388096]
        )
      ),
      linear2: LinearParameters(
        weight: Tensor<Float>(
          [
            [-0.04711317, 0.07374136],
            [-0.058497474, -0.011900932],
          ]
        ),
        bias: Tensor<Float>(
          [-0.008636497, 0.052838206]
        )
      )
    ),
    mlp_memory: MLPParameters(
      linear1: LinearParameters(
        weight: Tensor<Float>(
          [
            [-0.06895532, 0.0675631],
            [0.062555104, -0.006975107],
          ]
        ),
        bias: Tensor<Float>(
          [0.021453634, 0.0056577697]
        )
      ),
      linear2: LinearParameters(
        weight: Tensor<Float>(
          [
            [0.01337228, 0.07287796],
            [-0.025111437, -0.021482762],
            [-0.05161675, -0.06811503],
            [-0.072463006, 0.015226476],
          ]
        ),
        bias: Tensor<Float>(
          [0.0032938793, -0.043962937, 0.043240592, 0.0678826]
        )
      )
    ),
    emb_dec: EmbeddingParameters(
      weight: Tensor<Float>(
        [
          [0.055306703, 0.07086456],
          [0.03423354, -0.015132636],
          [-0.04077827, 0.016811028],
          [-0.037189033, -0.07027687],
          [0.054974243, 0.017300054],
        ]
      )
    ),
    lstm_dec: LSTMParameters(
      weight_ih_l0: Tensor<Float>(
        [
          [-0.053585358, -0.0642758],
          [-0.07246614, 0.025658146],
          [0.034285776, -0.014611781],
          [0.058412, 0.047652483],
          [0.065825045, 0.042562716],
          [0.050531074, 0.047255352],
          [-0.03512928, 0.004992813],
          [0.005484812, -0.054734543],
        ]
      ),
      weight_hh_l0: Tensor<Float>(
        [
          [0.07812595, 0.0031644031],
          [-0.04185462, 0.03933753],
          [-0.044581212, -0.018176649],
          [0.07533194, 0.0030083433],
          [-0.045243938, -0.026109837],
          [0.046121553, -0.053141937],
          [0.011378422, 0.067420706],
          [-0.05194992, 0.044939123],
        ]
      ),
      bias_ih_l0: Tensor<Float>(
        [
          0.049315616, -0.05961135, -0.047641095, 0.056274325, -0.071667776,
          0.049188778, -0.05663743, -0.051864214,
        ]
      ),
      bias_hh_l0: Tensor<Float>(
        [
          0.055988565, 0.01968135, -0.057932526, 0.024752177, -0.029085837,
          -0.03911104, -0.0015038475, 0.051634952,
        ]
      )
    ),
    linear_dec: LinearParameters(
      weight: Tensor<Float>(
        [
          [-0.03417959, 0.04824567],
          [0.0559683, 0.0076355636],
          [-0.03857645, 0.015529476],
          [0.057112962, 0.036605842],
          [0.023432933, -0.023976203],
        ]
      ),
      bias: Tensor<Float>(
        [-0.03640049, -0.057923757, 0.05912192, 0.03688284, 0.06261988]
      )
    )
  )
  static let expectedEncoding = Tensor<Float>(
    [
      [-0.016786836, 0.0014875316],
      [-0.024643049, 0.003509362],
      [-0.030508908, 0.005803111],
      [-0.031535156, 0.0056067896],
    ]
  )
  static let expectedMLPInterpolationOutput = Tensor<Float>(
    [
      [-0.72172225, -0.665366],
      [-0.7217337, -0.6653552],
      [-0.72174466, -0.66534483],
      [-0.7217447, -0.6653447],
    ]
  )
  static let expectedMLPMemoryOutput = Tensor<Float>(
    [
      [-1.400075, -1.4486395, -1.3622522, -1.3377004],
      [-1.4000794, -1.4486222, -1.3622293, -1.3377339],
      [-1.4000806, -1.4486088, -1.3622129, -1.3377609],
      [-1.4000823, -1.448607, -1.3622097, -1.3377641],
    ]
  )
  static let expectedDecoded = Tensor<Float>(
    [-6.5631247, -4.930211]
  )
  static let lattice = Lattice(
    positions: [
      Lattice.Node(
        bestEdge: nil,
        bestScore: 0.0,
        edges: [],
        semiringScore: SemiRing(logp: 0.0, logr: -Float.infinity)
      ),
      Lattice.Node(
        bestEdge: nil,
        bestScore: 0.0,
        edges: [
          Lattice.Edge(
            start: 0,
            end: 1,
            string: CharacterSequence(_debug: 1),
            logp: -3.9122378826141357,
            score: SemiRing(logp: -3.9122378826141357, logr: -3.9122378826141357),
            totalScore: SemiRing(logp: -3.9122378826141357, logr: -3.9122378826141357)
          )
        ],
        semiringScore: SemiRing(logp: -3.9122378826141357, logr: -3.9122378826141357)
      ),
      Lattice.Node(
        bestEdge: nil,
        bestScore: 0.0,
        edges: [
          Lattice.Edge(
            start: 0,
            end: 2,
            string: CharacterSequence(_debug: 1),
            logp: -2.0545620918273926,
            score: SemiRing(logp: -2.0545620918273926, logr: 1.4111738204956055),
            totalScore: SemiRing(logp: -2.0545620918273926, logr: 1.4111738204956055)
          ),
          Lattice.Edge(
            start: 1,
            end: 2,
            string: CharacterSequence(_debug: 1),
            logp: -3.936295986175537,
            score: SemiRing(logp: -3.936295986175537, logr: -3.936295986175537),
            totalScore: SemiRing(logp: -7.848533630371094, logr: -7.155386447906494)
          ),
        ],
        semiringScore: SemiRing(logp: -2.051520824432373, logr: 1.411364197731018)
      ),
      Lattice.Node(
        bestEdge: nil,
        bestScore: 0.0,
        edges: [
          Lattice.Edge(
            start: 0,
            end: 3,
            string: CharacterSequence(_debug: 1),
            logp: -7.253466606140137,
            score: SemiRing(logp: -7.253466606140137, logr: -1.7604050636291504),
            totalScore: SemiRing(logp: -7.253466606140137, logr: -1.7604050636291504)
          ),
          Lattice.Edge(
            start: 1,
            end: 3,
            string: CharacterSequence(_debug: 1),
            logp: -2.0307273864746094,
            score: SemiRing(logp: -2.0307273864746094, logr: 1.4350085258483887),
            totalScore: SemiRing(logp: -5.942965507507324, logr: -2.446457624435425)
          ),
          Lattice.Edge(
            start: 2,
            end: 3,
            string: CharacterSequence(_debug: 1),
            logp: -3.912229061126709,
            score: SemiRing(logp: -3.912229061126709, logr: -3.912229061126709),
            totalScore: SemiRing(logp: -5.963749885559082, logr: -2.4700069427490234)
          ),
        ],
        semiringScore: SemiRing(logp: -5.1324286460876465, logr: -1.0695605278015137)
      ),
      Lattice.Node(
        bestEdge: nil,
        bestScore: 0.0,
        edges: [
          Lattice.Edge(
            start: 0,
            end: 4,
            string: CharacterSequence(_debug: 1),
            logp: -8.936973571777344,
            score: SemiRing(logp: -8.936973571777344, logr: -2.0055017471313477),
            totalScore: SemiRing(logp: -8.936973571777344, logr: -2.0055017471313477)
          ),
          Lattice.Edge(
            start: 1,
            end: 4,
            string: CharacterSequence(_debug: 1),
            logp: -7.277979850769043,
            score: SemiRing(logp: -7.277979850769043, logr: -1.7849183082580566),
            totalScore: SemiRing(logp: -11.190217971801758, logr: -5.69304895401001)
          ),
          Lattice.Edge(
            start: 2,
            end: 4,
            string: CharacterSequence(_debug: 1),
            logp: -2.0545456409454346,
            score: SemiRing(logp: -2.0545456409454346, logr: 1.4111902713775635),
            totalScore: SemiRing(logp: -4.106066703796387, logr: 0.051392197608947754)
          ),
          Lattice.Edge(
            start: 3,
            end: 4,
            string: CharacterSequence(_debug: 1),
            logp: -3.936281204223633,
            score: SemiRing(logp: -3.936281204223633, logr: -3.936281204223633),
            totalScore: SemiRing(logp: -9.068710327148438, logr: -4.98878812789917)
          ),
        ],
        semiringScore: SemiRing(logp: -4.090378284454346, logr: 0.1802457720041275)
      ),
    ]
  )
  static let gradWrtLogR = SNLMParameters(
    emb_enc: EmbeddingParameters(
      weight: Tensor<Float>(
        [
          [-1.0885849e-05, 2.5600420e-05],
          [-2.0048559e-05, 5.2774609e-05],
          [-2.1089169e-05, 6.0592978e-05],
          [0.0000000e+00, 0.0000000e+00],
          [0.0000000e+00, 0.0000000e+00],
        ]
      )
    ),
    lstm_enc: LSTMParameters(
      weight_ih_l0: Tensor<Float>(
        [
          [-5.5954405e-07, 2.1259700e-07],
          [-9.2493622e-08, 1.2687329e-07],
          [4.8279912e-07, -5.5000936e-07],
          [-1.1893751e-07, 9.6952142e-08],
          [1.7744465e-05, -6.3974912e-06],
          [2.3670214e-05, -6.8028967e-06],
          [6.9926745e-07, 1.0588951e-07],
          [-3.6788197e-07, 1.1367500e-07],
        ]
      ),
      weight_hh_l0: Tensor<Float>(
        [
          [-6.1557824e-07, 7.9759928e-08],
          [1.8759494e-07, -2.4723642e-08],
          [-3.8983197e-07, 5.1562441e-08],
          [7.6582531e-08, -1.0336059e-08],
          [1.6366921e-05, -2.1041824e-06],
          [2.5587158e-05, -3.2796638e-06],
          [-7.9833825e-07, 1.1357896e-07],
          [2.2801314e-07, -3.2397072e-08],
        ]
      ),
      bias_ih_l0: Tensor<Float>(
        [
          5.0008435e-05, -1.0976097e-05, 1.7231478e-05, -3.3032948e-06,
          -1.3631615e-03, -2.0683720e-03, 5.0045674e-05, -1.1407620e-05,
        ]
      ),
      bias_hh_l0: Tensor<Float>(
        [
          5.0008435e-05, -1.0976097e-05, 1.7231478e-05, -3.3032948e-06,
          -1.3631615e-03, -2.0683720e-03, 5.0045674e-05, -1.1407620e-05,
        ]
      )
    ),
    mlp_interpolation: MLPParameters(
      linear1: LinearParameters(
        weight: Tensor<Float>(
          [
            [-2.162833e-04, 3.404662e-05],
            [-1.626215e-03, 2.559960e-04],
          ]
        ),
        bias: Tensor<Float>(
          [0.008965784, 0.06741227]
        )
      ),
      linear2: LinearParameters(
        weight: Tensor<Float>(
          [
            [0.03779146, 0.041938413],
            [-0.03779146, -0.041938413],
          ]
        ),
        bias: Tensor<Float>(
          [0.7893658, -0.7893658]
        )
      )
    ),
    mlp_memory: MLPParameters(
      linear1: LinearParameters(
        weight: Tensor<Float>(
          [
            [0.0006783868, -0.00010455749],
            [0.002728255, -0.00042056653],
          ]
        ),
        bias: Tensor<Float>(
          [-0.028684907, -0.11537111]
        )
      ),
      linear2: LinearParameters(
        weight: Tensor<Float>(
          [
            [-0.0098278085, -0.0017497203],
            [-0.009362103, -0.0016668034],
            [0.029616904, 0.005273029],
            [-0.010426991, -0.0018565056],
          ]
        ),
        bias: Tensor<Float>(
          [-0.4213174, -0.40135252, 1.2696781, -0.44700825]
        )
      )
    ),
    emb_dec: EmbeddingParameters(
      weight: Tensor<Float>(
        [
          [2.8222625e-04, 1.9048888e-04],
          [1.8905671e-04, 1.5085124e-04],
          [0.0000000e+00, 0.0000000e+00],
          [3.5629273e-06, 2.5625954e-05],
          [0.0000000e+00, 0.0000000e+00],
        ]
      )
    ),
    lstm_dec: LSTMParameters(
      weight_ih_l0: Tensor<Float>(
        [
          [-1.3549015e-05, -1.4535895e-05],
          [3.1118125e-07, -1.6063163e-07],
          [-9.2411565e-06, -8.3111609e-06],
          [3.1314161e-07, -9.0987783e-08],
          [2.9817267e-04, 3.1956812e-04],
          [2.4017896e-05, -1.0910597e-04],
          [-1.9980842e-05, -2.4779467e-05],
          [1.5680398e-07, -8.3687371e-07],
        ]
      ),
      weight_hh_l0: Tensor<Float>(
        [
          [7.7975146e-06, -7.3286355e-07],
          [-4.4332864e-07, 4.7302478e-08],
          [7.1966651e-06, -7.2089733e-07],
          [-3.4162869e-07, 4.1011177e-08],
          [-1.7509436e-04, 1.6362043e-05],
          [-1.0614634e-04, 1.1837798e-05],
          [9.3842154e-06, -8.1990322e-07],
          [-4.8998282e-07, 6.6971602e-08],
        ]
      ),
      bias_ih_l0: Tensor<Float>(
        [
          -1.8724482e-04, 1.3764327e-05, -1.8988468e-04, 8.8408688e-06,
          4.2812643e-03, 3.6167959e-03, -2.0367328e-04, 1.3456454e-05,
        ]
      ),
      bias_hh_l0: Tensor<Float>(
        [
          -1.8724482e-04, 1.3764327e-05, -1.8988468e-04, 8.8408688e-06,
          4.2812643e-03, 3.6167959e-03, -2.0367328e-04, 1.3456454e-05,
        ]
      )
    ),
    linear_dec: LinearParameters(
      weight: Tensor<Float>(
        [
          [-0.0041010594, 0.00012692134],
          [-0.005898418, 0.0008138743],
          [0.006042951, -0.0006127681],
          [-0.0020920308, 0.00028523407],
          [0.0060485564, -0.0006132616],
        ]
      ),
      bias: Tensor<Float>(
        [0.14542846, 0.14904119, -0.16054428, 0.026781656, -0.16070703]
      )
    )
  )
}
