import numpy as np
import icp
import datatools
import matplotlib.pyplot as plt


if __name__ == "__main__":
   
    # Load pre-processed model point cloud
    print("Extracting MODEL object...")
    model = datatools.load_XYZ_data_to_vec('data/data01_object.xyz')[:,:3]
    
    # Load raw data point cloud
    print("Extracting DATA02 object...")
    data02_object = datatools.load_XYZ_data_to_vec('data/data02_object.xyz')
    
    # Load raw data point cloud
    print("Extracting DATA03 object...")
    data03_object = datatools.load_XYZ_data_to_vec('data/data03_object.xyz')

    ref = model
    data = data02_object        # Here to test qualitycheck with the flawless model
    # data = data03_object      # Here uncomment to test qualitycheck with the misshapen model
    
    print('Reference size : ' + str(ref.shape))
    print('Raw data  size : ' + str(data.shape))
    


    ##########################################################################
    # Call ICP:
    #   Here you have to call the icp function in icp library, get its return
    #   variables and apply the transformation to the model in order to overlay
    #   it onto the reference model.

    matrix = np.eye(4,4)             # Transformation matrix returned by icp function
    errors = np.zeros((1,100))  # Error value for each iteration of ICP
    iterations = 100            # The total number of iterations applied by ICP
    total_time = 0                # Total time of convergence of ICP

    # ------- YOUR TURN HERE -------- 

    matrix, errors, it, total_time = icp.icp(data, ref)

    # ------- YOUR TURN HERE -------- 
    
    # Draw results
    fig = plt.figure(1, figsize=(20, 5))
    ax = fig.add_subplot(131, projection='3d')
    # Draw reference
    datatools.draw_data(ref, title='Reference', ax=ax)
    
    ax = fig.add_subplot(132, projection='3d')
    # Draw original data and reference
    datatools.draw_data_and_ref(data, ref=ref, title='Raw data', ax=ax)
    
    ##########################################################################
    # Apply transformation found with ICP to data:
    #
    # EXAMPLE of how to apply a homogeneous transformation to a set of points
    """
    # (1) Make a homogeneous representation of the model to transform
    homogeneous_model = np.ones((data.shape[0], 4))   ##### Construct a [N,4] matrix
    homogeneous_model[:,0:3] = np.copy(data)                   ##### Replace the X,Y,Z columns with the model points
    # (2) Construct the R|t homogeneous transformation matrix / here a rotation of 36 degrees around x axis
    theta = np.radians(36)
    c, s = np.cos(theta), np.sin(theta)
    homogeneous_matrix = np.array([[1, 0, 0, 0],
                                   [0, c, s, 0],
                                   [0, -s, c, 0],
                                   [0, 0, 0, 1]])
    # (3) Apply the transformation
    transformed_model = np.dot(homogeneous_matrix, homogeneous_model.T).T
    # (4) Remove the homogeneous coordinate
    transformed_model = np.delete(transformed_model, 3, 1)
    """

    # ------- YOUR TURN HERE -------- 

    homogeneous_data = np.ones((data.shape[0], 4))
    homogeneous_data[:,0:3] = np.copy(data)

    transformed_data = np.dot(matrix, homogeneous_data.T).T

    transformed_data = np.delete(transformed_data, 3, 1)
    

    ##########################################################################
    # Results display:
    #   Uncomment lines below and replace '...' with the correct variables (data, ref, errors, etc.)
    ax = fig.add_subplot(133, projection='3d')
    
    #### Draw transformed data and reference:
    datatools.draw_data_and_ref(transformed_data, ref, title='Registered data', ax=ax)
    
    #### Display error progress over time
    fig1 = plt.figure(2, figsize=(20,3))
    it = np.arange(0, len(errors), 1)
    plt.plot(it, errors)
    plt.ylabel('Residual distance')
    plt.xlabel('Iterations')
    plt.title('Total elapsed time :' + str(total_time) + ' s.')
    fig1.show()

    plt.show(block=True)

    np.savetxt("data/reference.xyz", ref, delimiter=" ")
    np.savetxt("data/data02.xyz", transformed_data, delimiter=" ")