#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


picked_objects = []

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


def vox_downsample(src_cloud):
    print("Applying VOX downsampling")
    vox = src_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    res_cloud = vox.filter()
    print("Vox grid Downsampled point cloud of length: {}".format(res_cloud.size))
    return res_cloud

def passthrough_filter(src_cloud):
    print("Applying passthough filter")

    passthrough = src_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.60
    axis_max = 1.1

    passthrough.set_filter_limits(axis_min, axis_max)
    res_cloud = passthrough.filter()

    # Passthrough filter on 'y'
    passthrough = res_cloud.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)

    axis_min = -0.5
    axis_max = 0.5

    passthrough.set_filter_limits(axis_min, axis_max)
    res_cloud = passthrough.filter()

#    pcl.save(out_cloud, "passthrough.pcd")    
    return res_cloud

def ransac_filter(src_cloud):

    print("Applying RANSAC Segmentation")

    seg = src_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()


    cloud_table = src_cloud.extract(inliers, negative=False)
    cloud_objects = src_cloud.extract(inliers, negative=True)

    print("Table Cloud points count: {}".format(cloud_table.size))
    print("Objects Cloud points count: {}".format(cloud_objects.size))

    return cloud_objects, cloud_table

def filter_outliers(src_cloud):
    print("Applying outliers filter")
    fil = src_cloud.make_statistical_outlier_filter()
    fil.set_mean_k(5)
    fil.set_std_dev_mul_thresh(0.5)
    res_cloud = fil.filter()
    print("Resulting point cloud size {}".format(res_cloud.size))
    fil.set_negative(True)
    
    return res_cloud

def find_cluster_indices(src_cloud):
    print("Clustering")
    white_cloud = XYZRGB_to_XYZ(src_cloud)
    tree = white_cloud.make_kdtree()


    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(2000)

    ec.set_SearchMethod(tree)


    cluster_indices = ec.Extract()
    color_cluster_point_list = []
    
    cluster_color = get_color_list(len(cluster_indices))

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])


    #Create new cloud containing all clusters, each with unique color ???????
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    ros_cluster_cloud = pcl_to_ros(cluster_cloud)   

    # TODO: Publish ROS messages
    pcl_cluster_pub.publish(ros_cluster_cloud)

    return cluster_indices





# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):


    print("pcl Callback invoked")
    cloud = ros_to_pcl(pcl_msg)
    cloud_filtered = vox_downsample(cloud)

    # PassThrough filter
    passthrough_filtered = passthrough_filter(cloud_filtered)

    #Filter noise
    outlier_filtered = filter_outliers(passthrough_filtered)

    # RANSAC plane segmentation
    cloud_objects, cloud_table = ransac_filter(outlier_filtered)

    #ros_cloud_objects = pcl_to_ros(cloud_objects)
    #pcl_objects_pub.publish(ros_cloud_objects)


    cluster_indices = find_cluster_indices(cloud_objects)
    print("Cluster indices count = {0}".format(len(cluster_indices)))


# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)

    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        print("Index: {0}, pts_list size: {1}".format(index, len(pts_list)))

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_pcl_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_pcl_cluster, using_hsv=True)
        normals = get_normals(ros_pcl_cluster)
        nhists = compute_normal_histograms(normals)

        feature = np.concatenate((chists, nhists))

        # Make the prediction
        print("Making prediction")
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]


        detected_objects_labels.append(label)

        label_pos = list(cloud_objects[pts_list[0]])
        label_pos[2] += .4

        # Publish a label into RViz
        print("Publising markers")
        object_markers_pub.publish(make_label(label,label_pos, index))
        

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_pcl_cluster
        detected_objects.append(do)



    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    pcl_objects_pub.publish(ros_cloud_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects, cloud_table)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list, cloud_table):

    # TODO: Initialize variables
    pick_labels = []
    pick_centroids = []
    
    test_scene_num = Int32()
    test_scene_num.data = 1
        
    dict_list = []
    
    yaml_filename = "output_{}.yaml".format(test_scene_num.data)
    
    
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    print("Object list size = {0}".format(len(object_list_param)))


    print("Iterating over object list")
    # TODO: Loop through the pick list
    for pick_object_param in object_list_param:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pick_object_name = pick_object_param['name']
        pick_object_group = pick_object_param['group']
        
        if(picked_objects.find(pick_object_name)):
            print("{0} has already been picked, moving on,.. ".format(pick_object_name))
            continue
        
        object_name = String()   
        print(pick_object_name.__class__)     
        object_name.data = str(pick_object_name)
        

        # Index of the object to be picked in the `detected_objects` list
        pick_object_cloud = None
        for i, detected_object in enumerate(object_list):
            if(detected_object.label == pick_object_name):
                pick_object_cloud = detected_object.cloud
            else:
                pass
            
        if(pick_object_cloud == None):
            print("error: {0} not found in object list".format(pick_object_name))
            continue
                            
            
        points_arr = ros_to_pcl(pick_object_cloud).to_array()

        pick_object_centroid = np.mean(points_arr, axis=0)[:3] 
        print("Centroid found : {0}".format(pick_object_centroid))

        pick_labels.append(pick_object_name)
        pick_centroids.append(pick_object_centroid)

        # Create pick_pose for the object
        pick_pose = Pose()
        pick_pose.position.x = float(pick_object_centroid[0])
        pick_pose.position.y = float(pick_object_centroid[1])
        pick_pose.position.z = float(pick_object_centroid[2])
                                                
        # TODO: Create 'place_pose' for the object
        place_pose = Pose()
        if(pick_object_group == 'green'):
            place_pose.position.x =  0
            place_pose.position.y = -0.71
            place_pose.position.z =  0.605
        else:
            place_pose.position.x =  0
            place_pose.position.y =  0.71
            place_pose.position.z =  0.605
                

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if(pick_object_group == 'green'):
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'
                    
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        
        dict_list.append(yaml_dict)
        
        '''
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        try:
            print("Creating service proxy,...")
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # TODO: Insert your message variables to be sent as a service request
            print("Requesting for service reponse,...")
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            #print ("Response: ",resp.success)
            picked_objects.append(pick_object_name)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''

    send_to_yaml(yaml_filename, dict_list)




if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)


    # TODO: Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model["classifier"]
    encoder = LabelEncoder()
    encoder.classes_ = model["classes"]
    scaler = model["scaler"]

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    rospy.spin()