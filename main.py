inputVector = [1.72, 1.23]
weights1 = [1.26, 0]
weights2 = [2.17, 0.32]

# Computing the dot product of inputVector and weights1
firstIndexesMult = inputVector[0] * weights1[0]
secondIndexesMult = inputVector[1] * weights1[1]
dotProduct1 = firstIndexesMult + secondIndexesMult

print(f"The dot product is: {dotProduct1}")
