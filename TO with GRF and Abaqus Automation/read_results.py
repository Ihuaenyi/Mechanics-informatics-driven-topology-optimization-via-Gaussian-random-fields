from odbAccess import *

odb = openOdb(path='Job-1.odb')
step = odb.steps['Step-1']

# Define the element set ROI_1
elementSet = odb.rootAssembly.elementSets['ROI']

# Extract strain and stress fields for the element set ROI_1
frame = step.frames[-1]

# Extract the strain fields 
strain = frame.fieldOutputs['PE']
strainValues = strain.getSubset(region=elementSet).values

# Write strain data to file
with open('strain.txt', 'w') as f:
    for v in strainValues:
        f.write('%.4e, %.4e, %.4e, %.4e, %.4e, %.4e \n' % (v.data[0], v.data[1], v.data[2], v.data[3], v.data[4], v.data[5]))

# Extract the stress fields 
stress = frame.fieldOutputs['S']
stressValues = stress.getSubset(region=elementSet).values

# Write stress data to file
with open('stress.txt', 'w') as f:
    for v in stressValues:
        f.write('%.4e, %.4e, %.4e, %.4e, %.4e, %.4e \n' % (v.data[0], v.data[1], v.data[2], v.data[3], v.data[4], v.data[5]))

# Close the odb
odb.close()
