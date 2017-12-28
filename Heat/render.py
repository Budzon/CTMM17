# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo, ArrayDatatype
import numpy as np
from obj_parser import ObjParser
from heat import MultiBodyHeat
import random

window_height, window_width = 400, 600

fov, aspect, z_near, z_far = 60.0, window_width / window_height, 0.1, 50.

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


def temps_to_grayscale(temps):
    temp_min = np.min(temps)
    temp_max = np.max(temps)
    return (temps + temp_min) / (temp_min + temp_max)


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


class ObjMesh(Mesh):
    def __init__(self, file_name, shader, part):
        super().__init__(shader)
        parser = ObjParser()
        parser.read_file(file_name)
        self.vertices = np.array(parser.vertices[part], dtype='f')
        self.indices = np.array(parser.indices[part], dtype='i')
        # self.colors = np.array([random.random() for _ in parser.vertices[part]], dtype='f')


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
    # meshes.append(Box([1., 1., 1.],
    #                   [0.0, 1.0, 0.0,
    #                    1.0, 0.0, 0.0,
    #                    1.0, 1.0, 0.0,
    #                    1.0, 1.0, 0.0,
    #                    1.0, 1.0, 0.0,
    #                    1.0, 1.0, 0.0], shader_attribute_color))
    # meshes.append(Cone(2.0, 2., 20, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0], shader_attribute_color))
    meshes.append(ObjMesh('model2bis.obj', shader_uniform_color, 0))
    meshes.append(ObjMesh('model2bis.obj', shader_uniform_color, 3))
    meshes.append(ObjMesh('model2bis.obj', shader_uniform_color, 4))
    meshes.append(ObjMesh('model2bis.obj', shader_uniform_color, 2))
    meshes.append(ObjMesh('model2bis.obj', shader_uniform_color, 1))
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


def set_uniforms():
    glUseProgram(shader_uniform_color)
    glUniform3fv(loc_color, 1, color)
    glUseProgram(0)


def display():
    global color
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glViewport(0, 0, window_width, window_height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClearDepth(1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    set_matrix_cube()
    temps = my_heat.propagate()
    print(temps)
    colors = temps_to_grayscale(temps)
    for ind, mesh in enumerate(meshes):
        color = np.array([colors[ind], colors[ind], colors[ind]])
        set_uniforms()
        # mesh.colors = np.array([col for _ in mesh.vertices], dtype='f')
        mesh.draw()

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


obj = ObjParser()
obj.read_file('model2bis.obj')
areas = obj.surface_areas()
num_of_parts = len(areas)

common_areas = np.zeros((num_of_parts, num_of_parts))
common_areas[0, 1] = common_areas[1, 0] = common_areas[1, 2] = common_areas[2, 1] = 4*np.pi
common_areas[2, 3] = common_areas[3, 2] = common_areas[3, 4] = common_areas[4, 3] = np.pi

conductivities = np.zeros((num_of_parts, num_of_parts))
conductivities[0, 1] = conductivities[1, 0] = 20
conductivities[1, 2] = conductivities[2, 1] = 130
conductivities[2, 3] = conductivities[3, 2] = 10.5
conductivities[3, 4] = conductivities[4, 3] = 119

blacknesses = np.array([5e-2, 5e-2, 1e-1, 1e-2, 1e-1])
cs = np.array([520, 520, 900, 840, 900])

inner_heats = [(lambda time: 0) for _ in range(num_of_parts)]
inner_heats[0] = lambda time: 1e-1 * (20 + 3*np.cos(time / 4))

my_heat = MultiBodyHeat(np.array([areas[0], areas[3], areas[4], areas[2], areas[1]]),
                        common_areas,
                        conductivities,
                        blacknesses,
                        cs,
                        inner_heats,
                        1, 2)
launch()
