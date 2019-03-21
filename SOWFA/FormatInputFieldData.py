import numpy as np
import sys
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

sys.path.append('/home/yluan/Documents/ML/TBNN/examples/turbulence')
from turbulencekepspreprocessor import TurbulenceKEpsDataProcessor
from turbulence_example_driver import plot_results
from tbnn import NetworkStructure, TBNN
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from Utilities import sampleData

"""
User Inputs
"""
caseName = 'ALM_N_H_ParTurb'
caseDir = '/media/yluan'
times = 'latest'
fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'gradUAvg', 'uuPrime2')
# Whether use existing pickle raw field data and/or ML x, y pickle data
useRawPickle, useXYpickle = True, False
# Whether save pickle fields
saveFields = True

# Whether confine the domain of interest, useful if mesh is too large
confineBox, plotConfinedBox = True, True
# Only when confineBox and saveFields are True
# Subscript of the confined field file name
confinedFieldNameSub = 'Confined'
# Whether auto generate confine box for each case,
# the confinement is bordered by 1st/2nd refinement zone
boxAutoDim = 'second'  # 'first', 'second', None
# Only when boxAutoDim is False
# Confine box counter-clockwise rotation in x-y plane
boxRot = np.pi/6
# Confine box origin, width, length, height
boxOrig = (0, 0, 0)  # (x, y, z)
boxL, boxW, boxH = 0, 0, 0  # float
# Absolute cap value for Sji and Rij and scalar basis x
capSijRij, capScalarBasis = 1e9, 1e9


"""
ML Settings
"""
# Whether to use sampling to reduce size of input data
sampling = True
# Only if sampling is True
# Number of samples and whether to use with replacement, i.e. same sample can be re-picked
sampleSize, replace = 10000, False
# Define parameters:
num_layers = 2  # Number of hidden layers in the TBNN
num_nodes = 20  # Number of nodes per hidden layer
max_epochs = 2000  # Max number of epochs during training
min_epochs = 1000  # Min number of training epochs required
interval = 100  # Frequency at which convergence is checked
average_interval = 4  # Number of intervals averaged over for early stopping criteria
split_fraction = 0.8  # Fraction of data to use for training
enforce_realizability = True  # Whether or not we want to enforce realizability constraint on Reynolds stresses
num_realizability_its = 5  # Number of iterations to enforce realizability
seed = 12345 # use for reproducibility, set equal to None for no seeding


"""
Process User Inputs
"""
inputsEnsembleName = 'Inputs_' + caseName
if times == 'latest':
    if caseName == 'ALM_N_H_ParTurb':
        times = '22000.0918025'

if confineBox and boxAutoDim is not None:
    if caseName == 'ALM_N_H_ParTurb':
        boxRot = np.pi/6
        # 1st refinement zone as confinement box
        if boxAutoDim == 'first':
            boxOrig = (1074.225, 599.464, 0)
            boxL, boxW, boxH = 1134, 1134, 405
        # 2nd refinement zone as confinement box
        elif boxAutoDim == 'second':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378, 216


"""
Read and Process Input Field Data
"""
# Initialize case
case = FieldData(caseName = caseName, caseDir = caseDir, times = times, fields = fields, save = saveFields)

# If not using existing pickle data, then read and process fields
if not useRawPickle:
    # Read field data
    fieldData = case.readFieldData()
    # Gradient of temporal mean U, nCell x 9
    # du/dx, du/dy, du/dz
    # dv/dx, dv/dy, dv/dz
    # dw/dx, dw/dy, dw/dz
    gradUAvg = fieldData['gradUAvg']
    # Get total temporal mean TKE, nCell
    kMean = fieldData['kResolved'] + fieldData['kSGSmean']
    # Get total temporal mean turbulence dissipation rate, nCell
    epsilonMean = case.calcMeanDissipationRateField(epsilonSGSmean = fieldData['epsilonSGSmean'], nuSGSmean = fieldData[
        'nuSGSmean'], resultPath = case.resultPath[times], save = saveFields)
    # Expand symmetric tensor to it's full form, nCell x 9
    # From xx, xy, xz
    #          yy, yz
    #              zz
    # to xx, xy, xz
    #    yx, yy, yz
    #    zx, zy, zz
    uuPrime2 = np.vstack((fieldData['uuPrime2'][:, 0], fieldData['uuPrime2'][:, 1], fieldData['uuPrime2'][:, 2],
                          fieldData['uuPrime2'][:, 1], fieldData['uuPrime2'][:, 3], fieldData['uuPrime2'][:, 4],
                          fieldData['uuPrime2'][:, 2], fieldData['uuPrime2'][:, 4], fieldData['uuPrime2'][:, 5])).T
    # Convert 1D array to 2D so that I can hstack them to 1 array, nCell x 1
    kMean, epsilonMean = kMean.reshape((-1, 1)), epsilonMean.reshape((-1, 1))
    # Horizontally stack these field data, nCell x 20
    # tke, epsilon, grad_u_00, grad_u_01, grad_u_02, grad_u_10, grad_u_11, grad_u_12, grad_u_20, grad_u_21, grad_u_22, uu_00, uu_01, uu_02, uu_10, uu_11, uu_12, uu_20, uu_21, uu_22
    inputsEnsemble = np.hstack((kMean, epsilonMean, fieldData['gradUAvg'], uuPrime2))
    # Save pickle if requested
    if saveFields:
        case.savePickleData(listData = inputsEnsemble, resultPath = case.resultPath[times], fileNames = inputsEnsembleName)

    # Read cell center coordinates, nCell and nCell x 3
    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()
    # Confine to domain of interest if requested
    if confineBox:
        ccx, ccy, ccz, cc, inputsEnsemble, box, flags = case.confineFieldDomain_Rotated(ccx, ccy, ccz, inputsEnsemble,
                                                                                    boxL = boxL, boxW = boxW, boxH = boxH, boxO = boxOrig, boxRot = boxRot,
                                                                                    save = saveFields, resultPath = case.resultPath[times], fileNameSub = confinedFieldNameSub, valsName = inputsEnsembleName)
        # Refresh new TKE, dissipation rate, grad(U), u'u' of the confined region
        kMean, epsilonMean = inputsEnsemble[:, 0], inputsEnsemble[:, 1]
        gradUAvg, uuPrime2 = inputsEnsemble[:, 2:11], inputsEnsemble[:, 11:]

        # Visualize the confined box if requested
        if plotConfinedBox:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            patch = patches.PathPatch(box, facecolor='orange', lw=0)
            ax.add_patch(patch)
            ax.axis('equal')
            ax.set_xlim(0, 3000)
            ax.set_ylim(0, 3000)
            # plt.show()

else:
    # If read whole field pickle data
    if not confineBox:
        cc = pickle.load(open(case.resultPath[times] + 'cc.p', 'rb'))
        inputsEnsemble = pickle.load(open(case.resultPath[times] + inputsEnsembleName + '.p', 'rb'))
    # Else if read confined field pickle data
    else:
        cc = pickle.load(open(case.resultPath[times] + 'cc_' + confinedFieldNameSub + '.p', 'rb'))
        inputsEnsemble = pickle.load(open(case.resultPath[times] + inputsEnsembleName + '_' + confinedFieldNameSub + '.p', 'rb'))

    kMean, epsilonMean = inputsEnsemble[:, 0], inputsEnsemble[:, 1]
    gradUAvg, uuPrime2 = inputsEnsemble[:, 2:11], inputsEnsemble[:, 11:]
    print('\nPickle raw inputs data read')

# Reshape tensors from nCell x 9 to nCell x 3 x 3
gradUAvg, uuPrime2 = gradUAvg.reshape((gradUAvg.shape[0], 3, 3)), uuPrime2.reshape((uuPrime2.shape[0], 3, 3))


"""
ML Train and Test
"""
# Set file names for pickle
fileNames = ('Sij', 'Rij', 'x', 'tb', 'y') if not confineBox \
    else \
    ('Sij_' + confinedFieldNameSub, 'Rij_' + confinedFieldNameSub, 'x_' + confinedFieldNameSub, 'tb_' + confinedFieldNameSub, 'y_' + confinedFieldNameSub)
# Calculate inputs and outputs
if not useXYpickle:
    data_processor = TurbulenceKEpsDataProcessor()
    Sij, Rij = data_processor.calc_Sij_Rij(gradUAvg, kMean, epsilonMean, cap = capSijRij)
    print('\nSij and Rij ready with |{}| cap'.format(capSijRij))
    x = data_processor.calc_scalar_basis(Sij, Rij, is_train = True, cap = capScalarBasis)  # Scalar basis
    print('\nInput scalar basis ready with |{}| cap'.format(capScalarBasis))
    tb = data_processor.calc_tensor_basis(Sij, Rij, quadratic_only=False)  # Tensor basis
    print('\nTensor basis ready')
    y = data_processor.calc_output(uuPrime2)  # Anisotropy tensor
    print('\nAnisotropy tensor ready')
    # Save files if requested
    if saveFields:
        case.savePickleData((Sij, Rij, x, tb, y), resultPath = case.resultPath[times], fileNames = fileNames)
# If use existing pickle data
else:
    dataDict = case.readPickleData(fileNames = fileNames, resultPath = case.resultPath[times])
    Sij, Rij = dataDict[fileNames[0]], dataDict[fileNames[1]]
    x, tb, y = dataDict[fileNames[2]], dataDict[fileNames[3]], dataDict[fileNames[4]]

# If use sampling
if sampling:
    (cc, x, tb, y) = sampleData((cc, x, tb, y), sampleSize = sampleSize, replace = replace)

# # Enforce realizability
# if enforce_realizability:
#     for i in range(num_realizability_its):
#         y = TurbulenceKEpsDataProcessor.make_realizable(y)
#
#     print('\nRealizability enforced on ground truth anisotropy tensor')

# Split into training and test data sets
if seed:
    np.random.seed(seed) # sets the random seed for Theano

x_train, tb_train, y_train, x_test, tb_test, y_test = \
    TurbulenceKEpsDataProcessor.train_test_split(x, tb, y, fraction=split_fraction, seed=seed)
print('\nTrain and test data split')

# Define network structure
structure = NetworkStructure()
structure.set_num_layers(num_layers)
structure.set_num_nodes(num_nodes)

# Initialize and fit TBNN
tbnn = TBNN(structure)
print('\nTBNN initialized with {0} hidden layers and {1} hidden nodes, start fitting...'.format(num_layers, num_nodes))
tbnn.fit(x_train, tb_train, y_train, max_epochs=max_epochs, min_epochs=min_epochs, interval=interval, average_interval=average_interval)

# Make predictions on train and test data to get train error and test error
labels_train = tbnn.predict(x_train, tb_train)
labels_test = tbnn.predict(x_test, tb_test)

# Enforce realizability
if enforce_realizability:
    for i in range(num_realizability_its):
        labels_train = TurbulenceKEpsDataProcessor.make_realizable(labels_train)
        labels_test = TurbulenceKEpsDataProcessor.make_realizable(labels_test)

# Determine error
rmse_train = tbnn.rmse_score(y_train, labels_train)
rmse_test = tbnn.rmse_score(y_test, labels_test)
print("RMSE on training data:", rmse_train)
print("RMSE on test data:", rmse_test)

# Plot the results
plot_results(y_test, labels_test)


