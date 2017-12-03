class ObjParser(object):
    def __init__(self):
        self.vertices = []
        self.indices = []

    def read_file(self, file_name):
        for line in open(file_name, 'r'):
            self.parse(line)

    def parse(self, line):
        entries = line.split()
        if entries[0] == "v":
            self.vertices += list(map(float, entries[1:4]))
        elif entries[0] == "f":
            quad = list(map(int, map(lambda s: s.split('/')[0], entries[1:])))
            if len(quad) == 3:
                self.indices += [quad[0]-1, quad[1]-1, quad[2]-1]
            elif len(quad) == 4:
                self.indices += [quad[0]-1, quad[1]-1, quad[2]-1, quad[2]-1, quad[3]-1, quad[0]-1]
