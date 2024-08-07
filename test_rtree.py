from rtree import index
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Function to create an R-tree index and find the bounding box containing key points
def find_enclosing_bbox(keypoints):
    # Create an R-tree index
    idx = index.Index()

    # Insert bounding boxes of key points into the R-tree
    for i, point in enumerate(keypoints):
        x, y = point
        bbox = (x, y, x, y)  # Bounding box with the same point as min and max
        idx.insert(i, bbox)

    # Perform a range query to find the bounding box containing key points
    enclosing_bbox = idx.bounds

    return enclosing_bbox

# # Example key points
# keypoints = [(2, 3), (5, 6), (8, 9), (10, 12), (3, 4), (7, 8)]

# # Find the bounding box containing key points
# result_bbox = find_enclosing_bbox(keypoints)


# print(result_bbox)
# # Plotting the points and the bounding box
# fig, ax = plt.subplots()

# # Plot the points
# x, y = zip(*keypoints)
# ax.scatter(x, y, color='blue', label='Key Points')

# # Plot the bounding box
# enclosing_rect = Rectangle((result_bbox[0], result_bbox[1]), 
#                            result_bbox[2] - result_bbox[0], 
#                            result_bbox[3] - result_bbox[1], 
#                            linewidth=2, edgecolor='red', facecolor='none', label='Enclosing Bounding Box')

# ax.add_patch(enclosing_rect)

# # Set axis limits
# ax.set_xlim(min(x)-1, max(x)+1)
# ax.set_ylim(min(y)-1, max(y)+1)

# # Add legend
# ax.legend()

# # Show the plot
# plt.show()
