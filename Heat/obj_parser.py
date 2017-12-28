import numpy as np


class ObjParser(object):
    def __init__(self):
        self.vertices = []
        self.indices = []
        self.num_of_parts = 0
        self.index_shift = 0

    def read_file(self, file_name):
        for line in open(file_name, 'r'):
            self.parse(line)

    def parse(self, line):
        entries = line.split()
        if entries[0] == "v":
            self.vertices[-1] += list(map(float, entries[1:4]))
        elif entries[0] == "f":
            quad = list(map(int, map(lambda s: s.split('/')[0], entries[1:])))
            if len(quad) == 3:
                self.indices[-1] += [quad[0]-1-self.index_shift, quad[1]-1-self.index_shift, quad[2]-1-self.index_shift]
            elif len(quad) == 4:
                self.indices[-1] += [quad[0]-1-self.index_shift, quad[1]-1-self.index_shift, quad[2]-1-self.index_shift, quad[2]-1-self.index_shift, quad[3]-1-self.index_shift, quad[0]-1-self.index_shift]
        elif entries[0] == "o":
            if self.num_of_parts != 0:
                self.index_shift += len(self.vertices[-1]) // 3
            self.num_of_parts += 1
            self.vertices.append([])
            self.indices.append([])

    def surface_areas(self):
        areas = []
        for part in range(self.num_of_parts):
            area = 0
            for triangle in range(len(self.indices[part]) // 3):
                ind1 = self.indices[part][3*triangle]
                ind2 = self.indices[part][3*triangle + 1]
                ind3 = self.indices[part][3*triangle + 2]
                p1 = np.array(self.vertices[part][3*ind1: 3*ind1 + 3])
                p2 = np.array(self.vertices[part][3*ind2: 3*ind2 + 3])
                p3 = np.array(self.vertices[part][3*ind3: 3*ind3 + 3])
                area += np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / 2
            areas += [area]
        return areas
