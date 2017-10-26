import tkinter as tk
from tkinter import ttk, colorchooser, filedialog

from matplotlib import figure, patches
from matplotlib.backends import backend_tkagg

import xml.etree.ElementTree as ET

class disc():

    def __init__(self, oX, oY, radius, colour):
        self.__oX = oX
        self.__oY = oY
        self.__radius = radius
        self.__colour = colour

    def getPatch(self):
        return patches.Circle((self.__oX, self.__oY), radius = self.__radius, color = self.__colour, fill = True)

    def toXML(self):
        return ET.Element('Disc', attrib = {'ox': '%f' % self.__oX, 
                                            'oy': '%f' % self.__oY, 
                                            'radius': '%f' % self.__radius,
                                            'colour': self.__colour})

class figureElement():

    def __init__(self, maxX = 100.):
        self.figure = figure.Figure(figsize=(4, 4), dpi=100)
        self.__plot = self.figure.add_subplot(111)
        self.__plot.grid(True)
        self.__maxX = maxX
        self.__setLimits()

    def __setLimits(self):
        self.__plot.set_xlim([-self.__maxX, self.__maxX])
        self.__plot.set_ylim([-self.__maxX, self.__maxX])

    def resize(self, factor):
        self.__maxX *= factor
        self.__setLimits()
        self.figure.canvas.draw()

    def limit(self):
        return self.__maxX

    def setEventReaction(self, eventName, reaction):
        self.figure.canvas.mpl_connect(eventName, reaction)

    def addAndDrawPatch(self, patch):
        self.figure.gca().add_patch(patch)
        self.figure.canvas.draw()

    def clear(self):
        self.__plot.clear()
        self.__plot.grid(True)

    def toXML(self):
        return ET.Element('Figure', attrib = {'limit': '%f' % self.__maxX})

class sliderElement():

    def __init__(self, masterElement, variable, minVal = 1.0, maxVal = 100., orient = 'vertical'):
        self.__slider = tk.Scale(masterElement, variable = variable, from_ = minVal, to = maxVal, orient = orient)

    def placeGrid(self, row, column):
        self.__slider.grid(row = row, column = column)

    def rescale(self, maxVal):
        self.__slider['to'] = maxVal
        self.__slider['from'] = maxVal / 100.

    def toXML(self):
        return ET.Element('Slider', attrib = {'maxval': '%f' % self.__slider['to']})

class radioButtonGroupElement():

    def __init__(self, masterElement, variable, values, texts):
        self.__radioButtons = []
        for vt in zip(values, texts):
            self.__radioButtons.append(tk.Radiobutton(masterElement, value = vt[0], text = vt[1], variable=variable))

    def placeGrid(self, row, column):
        for rb in self.__radioButtons:
            rb.grid(row, column)
    def placeGrid(self):
        for rb in self.__radioButtons:
            rb.grid()

class VoilaApp():

    def __init__(self, colour = '#ff0000'):
    # Base elements / tabs
        self.root = tk.Tk()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill = tk.BOTH)#, expand = True)
        
        self.editTab = ttk.Frame(self.root)
        self.notebook.add(self.editTab, text = 'Edit')
        self.modelTab = ttk.Frame(self.root)
        self.notebook.add(self.modelTab, text = 'Model')

    # Variables
        self.radius = tk.DoubleVar()
        self.colour = colour
        self.algorithm = tk.IntVar()

    # Setting up editTab
        self.saveButton = tk.Button(self.editTab, text = "Save", command = lambda: self.__saveXML())
        self.saveButton.grid(row = 0, column = 1)
        self.loadButton = tk.Button(self.editTab, text = "Load", command = lambda: self.__loadXML())
        self.loadButton.grid(row = 0, column = 2)

        self.__figEl = figureElement()
        self.canvas = backend_tkagg.FigureCanvasTkAgg(self.__figEl.figure, master = self.editTab)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row = 1, column = 1)

        self.zoomInButton = tk.Button(self.editTab, text = "+", command = lambda: self.__resize(1/1.5))
        self.zoomInButton.grid(row = 2, column = 1)
        self.zoomOutButton = tk.Button(self.editTab, text = "-", command = lambda: self.__resize(1.5))
        self.zoomOutButton.grid(row = 2, column = 2)

        self.slider = sliderElement(self.editTab, self.radius)
        self.slider.placeGrid(1, 2)
        self.sliderValueEntry=tk.Entry(self.editTab, textvariable = self.radius)
        self.sliderValueEntry.grid(row = 1, column = 3)

        self.colourButton = tk.Button(self.editTab, text = "Select colour", command = self.__selectColour)
        self.colourButton.grid(row = 3, column = 2)

        self.mouseX = tk.Label(self.editTab)
        self.mouseX.grid(row=4, column=1)
        self.mouseY = tk.Label(self.editTab)
        self.mouseY.grid(row=4, column=2)

    # Setting up modelTab
        self.integrationProcedureRadioButtonGroup = radioButtonGroupElement(self.modelTab, self.algorithm, 
            [0,1,2,3,4], ['scipy', 'verlet', 'verlet-threading', 'verlet-multiprocessing', 'verlet-opencl'])
        # self.integrationProcedureRadioButtonGroup.placeGrid(1, 1)
        self.integrationProcedureRadioButtonGroup.placeGrid()

    # Setting up events
        self.__figEl.setEventReaction('motion_notify_event', lambda event: self.__onMouseMovementEvent(event))

        self.discs = []
        self.__figEl.setEventReaction('button_press_event', lambda event: self.__onMouseClickEvent(event))

    def run(self):
        self.root.mainloop()

    def toXML(self):
        app = ET.Element('App', attrib = {'colour': self.colour})
        app.append(self.__figEl.toXML())
        app.append(self.slider.toXML())
        
        discsEl = ET.Element('Discs')
        for disc in self.discs:
            discsEl.append(disc.toXML())
        app.append(discsEl)

        return ET.ElementTree(app)

    def __resize(self, factor):
        self.__figEl.resize(factor)
        self.slider.rescale(self.__figEl.limit())

    def __selectColour(self):
        self.colour = colorchooser.askcolor()[1]

    def __addAndDrawDisc(self, x, y, rad, colour):
        self.discs.append(disc(x, y, rad, colour))
        self.__figEl.addAndDrawPatch(self.discs[-1].getPatch())

    def __onMouseMovementEvent(self, event):
        self.mouseX['text'] = event.xdata
        self.mouseY['text'] = event.ydata

    def __onMouseClickEvent(self, event):
        self.__addAndDrawDisc(event.xdata, event.ydata, self.radius.get(), self.colour)

    def __saveXML(self):
        name = filedialog.asksaveasfilename()
        with open(name, 'wb') as f:
            self.toXML().write(f, encoding='utf-8') 

    def __loadXML(self):
        name = filedialog.askopenfilename()
        with open(name, 'rt') as f:
            tree = ET.parse(f)
        root = tree.getroot()

        self.__figEl.clear()
        self.__figEl.resize(float(root[0].attrib['limit']) / self.__figEl.limit())

        self.colour = root.attrib['colour']
        self.slider.rescale(float(root[1].attrib['maxval']))

        self.discs = []
        for disc in list(root[2]):
            self.__addAndDrawDisc(float(disc.attrib['ox']), float(disc.attrib['oy']), float(disc.attrib['radius']), disc.attrib['colour'])

app = VoilaApp()
app.run()