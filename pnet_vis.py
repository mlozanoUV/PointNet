from re import M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from stl import mesh


def show(img, mask, ax):
    #"""
    ax[0].cla()
    #ax[0].imshow(img, cmap='gray')
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("image")
    ax[1].cla()
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("mask")
    #plt.show()
    plt.pause(0.1)


def visualize_points(point_cloud):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
        }
    )

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    try:
        ax.scatter(
            df["x"], df["y"], df["z"]
        )
    except IndexError:
        pass
    ax.legend()
    plt.show()

def visualize_data(point_cloud, labels, LABELS=[0,1]):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    COLORS =['green', 'red', 'blue']
    for index, label in enumerate(LABELS):
        c_df = df[df["label"] == label]
        _a = 1
        if index == 0 or index == 2 : _a = 0.05
        try:
            ax.scatter(
                #c_df["x"], c_df["y"], c_df["z"], label=label, alpha= 0.8, c=COLORS[index], s=2)
                c_df["x"], c_df["y"], c_df["z"], c=c_df["z"], s=2)
        except IndexError:
            pass
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_single_point_cloud(point_clouds, label_clouds, idx, LABELS):
    label_map = LABELS + ["none"]
    point_cloud = point_clouds[idx]
    label_cloud = label_clouds[idx]
    visualize_data(point_cloud, [label_map[np.argmax(label)] for label in label_cloud], LABELS)

def mask2mesh(cfile):
    reader = vtk.vtkNrrdReader()
    reader.SetFileName("/home/miguel/RL-AtriaSeg/Training Set/KSNYHUBHHUJTYJ14UQZR/laendo.nrrd")
    reader.Update()

    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(1)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below 400 to 0
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.ComputeNormalsOn()
    dmc.ComputeGradientsOn()
    dmc.Update()

    # To remain largest region
    confilter =vtk.vtkPolyDataConnectivityFilter()
    confilter.SetInputData(dmc.GetOutput())
    confilter.SetExtractionModeToLargestRegion()
    confilter.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetInputData(confilter.GetOutput())
    writer.SetFileName('aorta_seg.stl')
    writer.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(confilter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer) 
    renderer_interactor = vtk.vtkRenderWindowInteractor()
    renderer_interactor.SetRenderWindow(renderer_window)
    renderer.SetBackground(1,1,1)
    renderer_window.SetSize(1024, 768)

    renderer_interactor.Initialize()
    renderer_window.Render()
    renderer_interactor.Start()


def mcubes(m_imgs, show=False):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(m_imgs, 1)

    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]

    # Write the mesh to file "cube.stl"
    cube.save('cube.stl')

    if show:
        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh1 = Poly3DCollection(verts[faces])
        mesh1.set_edgecolor('k')
        ax.add_collection3d(mesh1)

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        ax.set_xlim(0, 100)  # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(0, 300)  # b = 10
        ax.set_zlim(0, 100)  # c = 16

        #plt.tight_layout()
        plt.show()

