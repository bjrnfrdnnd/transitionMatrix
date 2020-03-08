# encoding: utf-8

# (c) 2017-2019 Open Risk, all rights reserved
#
# TransitionMatrix is licensed under the Apache 2.0 license a copy of which is included
# in the source distribution of TransitionMatrix. This is notwithstanding any licenses of
# third-party software included in this distribution. You may not use this file except in
# compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.


"""
Example workflow using transitionMatrix to estimate a matrix from LendingClub data
Input data are in a special cohort format as the published datasets have some limitations

"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import transitionMatrix as tm
from transitionMatrix import source_path
from transitionMatrix.estimators import simple_estimator as es

dataset_path = source_path + "datasets/"

# Example: LendingClub Style Migration Matrix
# Load historical data into pandas frame
# Format:
# Expected Data Format is (ID, State_IN, State_OUT)

# Step 1
# Load the data set into a pandas frame
# Make sure state is read as a string and not as integer
print("Step 1")
data = pd.read_csv(dataset_path + 'LoanStats3a_Step2.csv')
# Data is in pandas frame, all pandas methods are available
print(data.describe())

# Step 2
# Describe and validate the State Space against the data
print("Step 2")
definition = [('A', "Grade A"), ('B', "Grade B"), ('C', "Grade C"),
               ('D', "Grade D"), ('E', "Grade E"), ('F', "Grade F"),
               ('G', "Grade G"), ('H', "Delinquent"), ('I', "Charged Off"),
               ('J', "Repaid")]
definition = [('A', "Grade A"), ('B', "Grade B"),K]
myState = tm.StateSpace(definition)
myState.describe()
labels = {'State': 'State_IN'}
print(myState.validate_dataset(dataset=data, labels=labels))
labels = {'State': 'State_OUT'}
print(myState.validate_dataset(dataset=data, labels=labels))

# Step 3
# Estimate matrices using Simple Estimator (Frequency count)
# compute confidence interval using goodman method at 95% confidence level

print("Step 3")
myEstimator = es.SimpleEstimator(states=myState, ci={'method': 'goodman', 'alpha': 0.05})
# resulting matrix array is returned as result
result = myEstimator.fit(data)
# confidence levels are stored with the estimator
myEstimator.summary()

# Step 4
# Review numerical results
print("Step 4")
myMatrix = tm.TransitionMatrix(result)
myMatrix.print()

print(myMatrix.validate())
print(myMatrix.characterize())
myMatrix.print()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

plt.style.use(['ggplot'])
plt.ylabel('From State')
plt.xlabel('To State')
mymap = plt.get_cmap("RdYlGn")
mymap = plt.get_cmap("Reds")
# mymap = plt.get_cmap("Greys")
normalize = mpl.colors.LogNorm(vmin=0.0001, vmax=1)

matrix_size = myMatrix.shape[0]
square_size = 1.0 / matrix_size

diagonal = myMatrix.diagonal()
# colors = []

ax.set_xticklabels(range(0, matrix_size))
ax.set_yticklabels(range(0, matrix_size))
ax.xaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))
ax.yaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))

# iterate over all elements of the matrix

for i in range(0, matrix_size):
    for j in range(0, matrix_size):
        if myMatrix[i, j] > 0:
            rect_size = np.sqrt(myMatrix[i, j]) * square_size
        else:
            rect_size = 0

        dx = 0.5 * (square_size - rect_size)
        dy = 0.5 * (square_size - rect_size)
        p = patches.Rectangle(
            (i * square_size + dx, j * square_size + dy),
            rect_size,
            rect_size,
            fill=True,
            color=mymap(normalize(myMatrix[i, j]))
        )
        ax.add_patch(p)

cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
cb = mpl.colorbar.ColorbarBase(cbax, cmap=mymap, norm=normalize, orientation='vertical')
cb.set_label("Transition Prabability", rotation=270, labelpad=15)

plt.show(block=True)
plt.interactive(False)