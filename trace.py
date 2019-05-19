class Operation:
	def __init__(self, name, left, right):
		self._name = name
		self._left = left
		self._right = right

	def cost(self, x, y, deltas):
		if(self._left >= 0):
			left = x[self._left]
		else:
			left = None
		if(self._right >= 0):
			right = y[self._right]
		else:
			right = None
		if(self._name):
			delta = deltas[self._name]
			return delta(left, right)
		else:
			return deltas(left, right)


	def render(self, x, y, deltas = None):
		op_str = ''
		if(self._name):
			op_str += str(self._name)
			op_str += ': '
		if(self._left >= 0):
			left = x[self._left]
			op_str += str(left)
		else:
			left = None
			op_str += '-'
		op_str += ' vs. '
		if(self._right >= 0):
			right = y[self._right]
			op_str += str(right)
		else:
			right = None
			op_str += '-'
		if(deltas):
			op_str += ': '
			if(self._name):
				delta = deltas[self._name]
				op_str += str(delta(left, right))
			else:
				op_str += str(deltas(left, right))
		return op_str

	def __repr__(self):
		op_str = ''
		if(self._name):
			op_str += str(self._name)
			op_str += ': '
		if(self._left >= 0):
			op_str += str(self._left)
		else:
			op_str += '-'
		op_str += ' vs. '
		if(self._right >= 0):
			op_str += str(self._right)
		else:
			op_str += '-'
		return op_str

	def __str__(self):
		return self.__repr__()

class Trace(list):
	def __init__(self):
		list.__init__(self, [])

	def append_operation(self, left, right, op = None):
		self.append(Operation(op, left, right))

	def cost(self, x, y, deltas):
		d = 0.
		for op in self:
			d += op.cost(x, y, deltas)
		return d

	def render(self, x, y, deltas = None):
		render =  []
		for op in self:
			render.append(op.render(x, y, deltas))
		return '\n'.join(render)
