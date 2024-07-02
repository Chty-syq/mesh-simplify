import numpy as np
import trimesh
from trimesh import Trimesh


def simplify(mesh: Trimesh, ratio=0.5, threshold=0):
    planes = np.array([get_plane_equation(*mesh.vertices[face]) for face in mesh.faces])
    q_matrices = np.array([get_q_matrix(mesh, planes, vid) for vid in range(len(mesh.vertices))])

    pairs = select_pairs(mesh, threshold)
    heap = sorted([(*optimize_pair(mesh, q_matrices, pair), pair) for pair in pairs], key=lambda x: x[0])

    v_status, f_status = np.ones(len(mesh.vertices)), np.ones(len(mesh.faces))
    target_nums = ratio * len(mesh.faces)
    while np.sum(f_status) > target_nums:
        _, v_opt, (v1, v2) = heap[0]
        # 更新顶点
        mesh.vertices[v1] = v_opt
        mesh.vertices[v2] = v_opt
        # 移除顶点v2
        v_status[v2] = 0
        # 移除v1,v2共有的面片
        v1_faces = np.where(mesh.faces == v1)[0]
        v2_faces = np.where(mesh.faces == v2)[0]
        common_faces = [f for f in v1_faces if f in v2_faces]  # 同时包含v1,v2的面片
        f_status[common_faces] = 0  # 删除共有面片
        # 更新面片
        mesh.faces[np.where(mesh.faces == v2)] = v1
        # 更新面片方程
        for idx in set(v1_faces) | set(v2_faces):
            planes[idx] = get_plane_equation(*mesh.vertices[mesh.faces[idx]]) if f_status[idx] else np.array([0, 0, 0, 0])
        # 更新顶点v1对应的Q矩阵
        q_matrices[[v1, v2]] = get_q_matrix(mesh, planes, v1)
        # 更新heap
        heap_new = []
        for item in heap:
            if v1 in item[2] or v2 in item[2]:
                u = item[2][0] if item[2][0] != v2 else v1
                v = item[2][1] if item[2][1] != v2 else v1
                pair = (min(u, v), max(u, v))
                if pair[0] != pair[1] and pair not in [rec[2] for rec in heap_new]:
                    cost, v_opt = optimize_pair(mesh, q_matrices, pair)
                    heap_new.append((cost, v_opt, pair))
            else:
                heap_new.append(item)

        heap = sorted(heap_new, key=lambda x: x[0])

    # 执行真删除
    v_serial = np.delete(np.arange(len(mesh.vertices)), np.where(v_status == 0)[0]).tolist()
    vertices = np.delete(mesh.vertices, np.where(v_status == 0)[0], axis=0)
    faces = np.delete(mesh.faces, np.where(f_status == 0)[0], axis=0)
    faces = np.array([[v_serial.index(p1), v_serial.index(p2), v_serial.index(p3)] for (p1, p2, p3) in faces])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def select_pairs(mesh: Trimesh, threshold):
    points = np.array(mesh.vertices)
    inner = np.matmul(points, points.T)
    xx = np.sum(points ** 2, axis=-1, keepdims=True)
    dist = xx - 2 * inner + xx.T
    pairs = [(vi, vj) for vi, vj in zip(*np.where(dist < threshold ** 2)) if vi < vj]
    edges = [(vi, vj) for vi, vj in zip(mesh.edges[:, 0], mesh.edges[:, 1])]
    pairs = pairs + edges + [(vj, vi) for vi, vj in edges]
    pairs = [(vi, vj) for vi, vj in pairs if vi < vj]
    pairs = sorted(list(set(pairs)))
    return pairs


def optimize_pair(mesh: Trimesh, q_matrices, pair):
    Q1, Q2 = q_matrices[list(pair)]
    Q = Q1 + Q2
    A = np.concatenate([Q[:3, :], [[0, 0, 0, 1]]], axis=0)
    if np.linalg.det(A) > 0:
        v_opt = np.linalg.inv(A) @ np.array([[0], [0], [0], [1]])
        cost = v_opt.T @ Q @ v_opt
        v_opt = v_opt[:3, 0]
    else:
        v1, v2 = mesh.vertices[list(pair)]
        vm = (v1 + v2) / 2
        v_opt = min([v1, v2, vm], key=lambda v: v.T @ Q @ v)
        cost = v_opt.T @ Q @ v_opt
        v_opt = v_opt[:3, 0]
    return float(cost), v_opt


def get_plane_equation(p1, p2, p3):
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(p1, normal)
    return np.array([*normal, d])


def get_q_matrix(mesh: Trimesh, planes, vid):
    q = np.zeros((4, 4))
    for k in np.where(mesh.faces == vid)[0]:
        p = planes[k].reshape(1, -1)
        q += p.T @ p
    return q


def main():
    mesh = trimesh.load("dinosaur.obj")
    # mesh.show()
    mesh = simplify(mesh, ratio=0.1, threshold=1)
    mesh.show()


if __name__ == '__main__':
    main()
