class Node:
    def evaluate(self):
        raise NotImplementedError()

class ConstantNumberNode(Node):
    def __init__(self, lex):
        self.lex = lex
        self.value = float(lex)
        
    def evaluate(self):
        # Insert your code here!!!
        return self.value
        

class BinaryNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
        
    def evaluate(self):
        lvalue = self.left.evaluate()
        rvalue = self.right.evaluate()
        return self.operate(lvalue, rvalue)
    
    @staticmethod
    def operate(lvalue, rvalue):
        raise NotImplementedError()
        

class PlusNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        # Insert your code here!!!
        return lvalue + rvalue
        

class MinusNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        # Insert your code here!!!
        return lvalue - rvalue

class StarNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        # Insert your code here!!!
        return lvalue * rvalue

class DivNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        # Insert your code here!!!
        return lvalue / rvalue