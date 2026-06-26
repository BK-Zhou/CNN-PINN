# abaqus_export.py
# 在 Abaqus/CAE 中 File -> Run Script 运行
from odbAccess import *
from abaqusConstants import *
import numpy as np

odb_path = 'Job-1.odb'  # 修改为你的 ODB 文件名
output_npz = 'abaqus_raw_data.npz'

odb = openOdb(path=odb_path)
assembly = odb.rootAssembly

# ===== 修正点 =====
# 原代码错误：instance = assembly.instances[instance.keys()[0]]
# 正确写法：先获取 instances 字典，再取第一个 key
instance_dict = assembly.instances
instance_name = instance_dict.keys()[0]  # 取第一个 instance 的名称
instance = instance_dict[instance_name]
# =================

# 提取节点坐标
nodes = instance.nodes
coords = np.array([[n.coordinates[0], n.coordinates[1]] for n in nodes])  # (N, 2)

# 提取时间步与温度
frames = odb.steps[odb.steps.keys()[0]].frames
n_frames = len(frames)
n_nodes = len(nodes)

times = np.zeros(n_frames)
temps = np.zeros((n_frames, n_nodes))  # (T, N)

for i, frame in enumerate(frames):
    times[i] = frame.frameValue
    field = frame.fieldOutputs['NT11']

    # 节点场值提取（处理节点编号映射）
    node_vals = np.zeros(n_nodes)
    # 建立 nodeLabel -> index 映射，防止编号不连续
    node_map = {n.label: idx for idx, n in enumerate(nodes)}

    for val in field.values:
        if val.nodeLabel in node_map:
            idx = node_map[val.nodeLabel]
            node_vals[idx] = val.data

    temps[i, :] = node_vals

np.savez(output_npz, coords=coords, temps=temps, times=times)
odb.close()
print("导出完成: %s, 节点数=%d, 时间步=%d" % (output_npz, n_nodes, n_frames))