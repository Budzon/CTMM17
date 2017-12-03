# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo, ArrayDatatype
import numpy as np
from obj_parser import ObjParser
import random

window_height, window_width = 400, 600

fov, aspect, z_near, z_far = 60.0, window_width / window_height, 0.1, 10.

angle_speed, mouse_speed, move_speed = 5, 5e-3, 5.
angles = np.zeros(2)

eye = np.array([0.0, 0.0, -2.0])
target = np.array([0.0, 0.0, 1.0])
up = np.array([0.0, 1.0, 0.0])
MAX_EYE_DIST = 25.0

global color, loc_color
global shader_uniform_color, shader_attribute_color
global meshes

move, left, right, forward, backward, upward, downward = 0, 1, 2, 4, 8, 16, 32


class Matrices:
    @staticmethod
    def translate(to):
        return np.array([
            [1, 0, 0, to[0]],
            [0, 1, 0, to[1]],
            [0, 0, 1, to[2]],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotate(by, v):
        s, c = np.sin(by), np.cos(by)
        axis = Matrices.normalize(v)
        tmp = (1-c)*axis

        rot = np.eye(4)
        rot[0, 0] = c + tmp[0]*axis[0]
        rot[1, 0] = tmp[0]*axis[1] + s*axis[2]
        rot[2, 0] = tmp[0]*axis[2] - s*axis[1]

        rot[0, 1] = tmp[1]*axis[0] - s*axis[2]
        rot[1, 1] = c + tmp[1]*axis[1]
        rot[2, 1] = tmp[1]*axis[2] + s*axis[0]

        rot[0, 2] = tmp[2]*axis[0] + s*axis[1]
        rot[1, 2] = tmp[2]*axis[1] - s*axis[0]
        rot[2, 2] = c + tmp[2]*axis[2]

        return rot

    @staticmethod
    def scale(by):
        return np.diagflat([by, 1])

    @staticmethod
    def normalize(v):
        m = np.linalg.norm(v)
        if m == 0:
            return v
        else:
            return v / m

    @staticmethod
    def look_at(eye, center, up):
        f = Matrices.normalize(center - eye)
        s = Matrices.normalize(np.cross(f, up))
        u = np.cross(Matrices.normalize(s), f)

        lookat = np.eye(4)
        # lookat[0, 0] = s[0]
        # lookat[0, 1] = s[1]
        # lookat[0, 2] = s[2]

        # lookat[1, 0] = u[0]
        # lookat[1, 1] = u[1]
        # lookat[1, 2] = u[2]
        #
        # lookat[2, 0] = -f[0]
        # lookat[2, 1] = -f[1]
        # lookat[2, 2] = -f[2]

        lookat[0, :3] = s
        lookat[1, :3] = u
        lookat[2, :3] = -f

        # lookat[0, 3] = -s.dot(eye)
        # lookat[1, 3] = -u.dot(eye)
        # lookat[2, 3] = f.dot(eye)

        return lookat.dot(Matrices.translate(-eye))

    @staticmethod
    def perspective(fovy, aspect, z_near, z_far):
        tan_half_fovy = np.tan(fovy * np. pi / 180 / 2)
        result = np.zeros((4, 4))
        result[0, 0] = 1 / aspect / tan_half_fovy
        result[1, 1] = 1 / tan_half_fovy
        result[2, 2] = (z_near + z_far) / (z_near - z_far)
        result[3, 2] = -1
        result[2, 3] = 2 * z_far * z_near / (z_near - z_far)

        return result


class Mesh:
    def __init__(self, shader):
        self.vao = -1
        self.vbo = []
        self.vertices = np.empty(1)
        self.indices = np.empty(1)
        self.data_count = 0
        self.shader = shader
        # self.colors = np.empty(1)

    def set_buffers(self):
        color_as_attribute = hasattr(self, 'colors')

        if self.vao == -1:
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(3 if color_as_attribute else 2)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.vertices), self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(glGetAttribLocation(self.shader, 'pos'), 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # glEnable(GL_ELEMENT_ARRAY_BUFFER)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo[1])
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices)*4, (ctypes.c_uint * len(self.indices))(*self.indices), GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.indices), self.indices, GL_STATIC_DRAW)

        if color_as_attribute:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[2])
            glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(self.colors), self.colors, GL_STATIC_DRAW)
            glVertexAttribPointer(glGetAttribLocation(self.shader, 'color'), 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def draw(self):
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)


class Box(Mesh):
    def __init__(self, half_sizes, colors, shader):
        super().__init__(shader)
        self.vertices = np.array([
            # Bottom
            half_sizes[0], -half_sizes[1], -half_sizes[2],
            half_sizes[0],  half_sizes[1], -half_sizes[2],
            -half_sizes[0],  half_sizes[1], -half_sizes[2],
            -half_sizes[0], -half_sizes[1], -half_sizes[2],
            # Top
            half_sizes[0], -half_sizes[1],  half_sizes[2],
            half_sizes[0], half_sizes[1],  half_sizes[2],
            -half_sizes[0], half_sizes[1],  half_sizes[2],
            -half_sizes[0], -half_sizes[1],  half_sizes[2],
            # Front
            half_sizes[0], -half_sizes[1], -half_sizes[2],
            half_sizes[0], half_sizes[1], -half_sizes[2],
            half_sizes[0], half_sizes[1], half_sizes[2],
            half_sizes[0], -half_sizes[1], half_sizes[2],
            # Back
            -half_sizes[0], -half_sizes[1], -half_sizes[2],
            -half_sizes[0], half_sizes[1], -half_sizes[2],
            -half_sizes[0], half_sizes[1], half_sizes[2],
            -half_sizes[0], -half_sizes[1], half_sizes[2],
            # Left
            half_sizes[0], -half_sizes[1], -half_sizes[2],
            -half_sizes[0], -half_sizes[1], -half_sizes[2],
            -half_sizes[0], -half_sizes[1], half_sizes[2],
            half_sizes[0], -half_sizes[1], half_sizes[2],
            # Right
            half_sizes[0], half_sizes[1], -half_sizes[2],
            -half_sizes[0], half_sizes[1], -half_sizes[2],
            -half_sizes[0], half_sizes[1], half_sizes[2],
            half_sizes[0], half_sizes[1], half_sizes[2]
        ], dtype='f')
        self.indices = np.array([
            # Bottom
            0, 1, 2, 2, 3, 0,
            # Top
            4, 5, 6, 6, 7, 4,
            # Front
            8, 9, 10, 10, 11, 8,
            # Back
            12, 13, 14, 14, 15, 12,
            # Left
            16, 17, 18, 18, 19, 16,
            # Right
            20, 21, 22, 22, 23, 20
        ], dtype='i')
        self.colors = np.array([
            # Top
            colors[0], colors[1], colors[2],
            colors[0], colors[1], colors[2],
            colors[0], colors[1], colors[2],
            colors[0], colors[1], colors[2],
            # Bottom
            colors[3], colors[4], colors[5],
            colors[3], colors[4], colors[5],
            colors[3], colors[4], colors[5],
            colors[3], colors[4], colors[5],
            # Front
            colors[6], colors[7], colors[8],
            colors[6], colors[7], colors[8],
            colors[6], colors[7], colors[8],
            colors[6], colors[7], colors[8],
            # Back
            colors[9], colors[10], colors[11],
            colors[9], colors[10], colors[11],
            colors[9], colors[10], colors[11],
            colors[9], colors[10], colors[11],
            # Left
            colors[12], colors[13], colors[14],
            colors[12], colors[13], colors[14],
            colors[12], colors[13], colors[14],
            colors[12], colors[13], colors[14],
            # Right
            colors[15], colors[16], colors[17],
            colors[15], colors[16], colors[17],
            colors[15], colors[16], colors[17],
            colors[15], colors[16], colors[17],
        ], dtype='f')


class Square(Mesh):
    def __init__(self, sizes, shader):
        super().__init__(shader)
        self.vertices = np.array([
            sizes[0], sizes[1], 0,
            sizes[0], -sizes[1], 0,
            -sizes[0], -sizes[1], 0,
            -sizes[0], -sizes[1], 0,
            -sizes[0], sizes[1], 0,
            sizes[0], sizes[1], 0
        ], dtype='f')
        self.indices = np.array([
            0, 1, 2, 3, 4, 5
        ], dtype='i')
        self.colors = np.array([
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0
        ], dtype='f')


class Cone(Mesh):
    def __init__(self, height, rad, slices, colors, shader):
        super().__init__(shader)

        vert = [0, 0, height]
        col = colors[:3]
        for slice in range(slices):
            ang = np.pi * (2*slice + 1) / slices
            vert += [rad*np.cos(ang), rad*np.sin(ang), 0]
            col += colors[:3]

        vert += [0, 0, 0]
        col += colors[3:]
        for slice in range(slices):
            ang = np.pi * (2*slice + 1) / slices
            vert += [rad*np.cos(ang), rad*np.sin(ang), 0]
            col += colors[3:]

        ind = []
        for slice in range(slices):
            ind += [0, 1 + slice, 1 + (1 + slice) % slices]
        for slice in range(slices):
            ind += [1 + slices, 1 + slices + (1 + slice), 2 + slices + (1 + slice) % slices]

        self.vertices = np.array(vert, dtype='f')
        self.indices = np.array(ind, dtype='i')
        self.colors = np.array(col, dtype='f')


class ObjMesh(Mesh):
    def __init__(self, file_name, shader):
        super().__init__(shader)
        parser = ObjParser()
        parser.read_file(file_name)
        self.vertices = np.array(parser.vertices, dtype='f')
        self.indices = np.array(parser.indices, dtype='i')
        self.colors = np.array([random.random() for _ in parser.vertices], dtype='f')


def init():
    global shader_uniform_color
    global shader_attribute_color

    vertex_uniform_color = """
        //uniform mat4 mvp;
    
        attribute vec3 pos;
        
        void main()
        {
            gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0); //gl_Vertex;
            //gl_Position = mvp * vec4(pos, 1.0);
        }
    """

    fragment_uniform_color = """
        uniform vec3 color;
    
        void main()
        {
            gl_FragColor = vec4(color, 1.0);
        }
    """

    shader_uniform_color = compileProgram(
        compileShader(vertex_uniform_color, GL_VERTEX_SHADER),
        compileShader(fragment_uniform_color, GL_FRAGMENT_SHADER)
    )

    vertex_attribute_color = """
            //uniform mat4 mvp;

            attribute vec3 pos;
            attribute vec3 color;
            
            varying vec4 f_color;
            
            void main()
            {
                gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0); //gl_Vertex;
                //gl_Position = mvp * vec4(pos, 1.0);
                f_color = vec4(color, 1.0);
            }
        """

    fragment_attribute_color = """
            varying vec4 f_color;

            void main()
            {
                gl_FragColor = f_color;
            }
        """

    shader_attribute_color = compileProgram(
        compileShader(vertex_attribute_color, GL_VERTEX_SHADER),
        compileShader(fragment_attribute_color, GL_FRAGMENT_SHADER)
    )

    global meshes
    meshes = []
    meshes.append(Box([1., 1., 1.],
                      [0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       1.0, 1.0, 0.0,
                       1.0, 1.0, 0.0,
                       1.0, 1.0, 0.0,
                       1.0, 1.0, 0.0], shader_attribute_color))
    meshes.append(Cone(2.0, 0.5, 20, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0], shader_attribute_color))
    meshes.append(ObjMesh('earth.obj', shader_attribute_color))
    for mesh in meshes:
        mesh.set_buffers()

    global color
    color = np.array([0.0, 0.8, 0.2])

    global loc_color
    loc_color = glGetUniformLocation(shader_uniform_color, 'color')


def reshape(w, h):
    global window_height, window_width
    window_height, window_width = h, w
    glViewport(0, 0, w, h)

    global aspect
    aspect = w / h
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect, z_near, z_far)
    glMatrixMode(GL_MODELVIEW)


def set_matrix_cube():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    cen = eye + target
    gluLookAt(eye[0], eye[1], eye[2], cen[0], cen[1], cen[2], up[0], up[1], up[2])


def set_matrix_cone():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    cen = eye + target
    gluLookAt(eye[0], eye[1], eye[2], cen[0], cen[1], cen[2], up[0], up[1], up[2])
    glTranslate(0.0, 0.0, 2.0)


def set_uniforms():
    glUseProgram(shader_uniform_color)
    glUniform3fv(loc_color, 1, color)
    glUseProgram(0)


def display():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glViewport(0, 0, window_width, window_height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClearDepth(1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    set_uniforms()

    set_matrix_cube()
    meshes[0].draw()
    meshes[2].draw()

    set_matrix_cone()
    meshes[1].draw()

    glutSwapBuffers()


def idle():
    if "prev_t" not in idle.__dict__:
        idle.prev_t = 0

    t = glutGet(GLUT_ELAPSED_TIME)
    dt = (t - idle.prev_t) * 1e-3
    idle.prev_t = t

    forward_dir = np.array([-np.sin(angles[0]), 0, np.cos(angles[0])])
    right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])

    global eye
    if move & left:
        eye -= right_dir * move_speed * dt
    if move & right:
        eye += right_dir * move_speed * dt
    if move & forward:
        eye += forward_dir * move_speed * dt
    if move & backward:
        eye -= forward_dir * move_speed * dt
    if move & upward:
        eye[1] += move_speed * dt
    if move & downward:
        eye[1] -= move_speed * dt

    eye = eye.clip(-MAX_EYE_DIST, MAX_EYE_DIST)

    glutPostRedisplay()


def keyboard(i_key, x, y):
    key = i_key.decode('utf-8')

    if key == 'q':
        exit(0)

    global move
    if key == 'w':
        move |= forward
    if key == 'a':
        move |= left
    if key == 's':
        move |= backward
    if key == 'd':
        move |= right
    if key == 'z':
        move |= upward
    if key == 'x':
        move |= downward


def keyboard_up(i_key, x, y):
    key = i_key.decode('utf-8')

    global move
    if key == 'w':
        move &= ~forward
    if key == 'a':
        move &= ~left
    if key == 's':
        move &= ~backward
    if key == 'd':
        move &= ~right
    if key == 'z':
        move &= ~upward
    if key == 'x':
        move &= ~downward


def motion(pos_x, pos_y):
    if "wrap" not in motion.__dict__:
        motion.wrap = False

    if not motion.wrap:
        window_size = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)

        global angles
        angles += mouse_speed * (np.array([pos_x, pos_y]) - np.array(window_size) / 2)

        if angles[0] < - np.pi:
            angles[0] += 2 * np.pi
        elif angles[0] > np.pi:
            angles[0] -= 2 * np.pi

        if angles[1] < -np.pi / 2 + 1e-3:
            angles[1] = -np.pi / 2 + 1e-3
        elif angles[1] > np.pi / 2 - 1e-3:
            angles[1] = np.pi / 2 - 1e-3

        global target
        target = np.array([
            -np.sin(angles[0]) * np.cos(angles[1]),
            -np.sin(angles[1]),
            np.cos(angles[0]) * np.cos(angles[1])
        ])
        motion.wrap = True
        glutWarpPointer(window_size[0] // 2, window_size[1] // 2)
    else:
        motion.wrap = False


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def launch():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(b"Hello")

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(motion)
    glutSetCursor(GLUT_CURSOR_NONE)

    try:
        init()
    except BaseException as e:
        print('Failed init: ' + str(e))
        exit(-1)

    try:
        glutMainLoop()
    except BaseException as e:
        print('Failed main loop: ' + str(e))


launch()
