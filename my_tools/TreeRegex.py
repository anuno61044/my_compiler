# import pydot
from cmp.utils import ContainerSet


class NFA:
    def __init__(self, states, finals, transitions, start=0):
        self.states = states
        self.start = start
        self.finals = set(finals)
        self.map = transitions
        self.vocabulary = set()
        self.transitions = { state: {} for state in range(states) }
        
        for (origin, symbol), destinations in transitions.items():
            assert hasattr(destinations, '__iter__'), 'Invalid collection of states'
            self.transitions[origin][symbol] = destinations
            self.vocabulary.add(symbol)
            
        self.vocabulary.discard('')

    def __getitem__(self,symbol:str,state=0):
        try:
            return self.map[state,symbol]
        except:
            print('No hay transiciones desde ese estado con ese símbolo')
            return None

    def epsilon_transitions(self, state):
        assert state in self.transitions, 'Invalid state'
        try:
            return self.transitions[state]['']
        except KeyError:
            return ()
            
    # def graph(self):
    #     G = pydot.Dot(rankdir='LR', margin=0.1)
    #     G.add_node(pydot.Node('start', shape='plaintext', label='', width=0, height=0))

    #     for (start, tran), destinations in self.map.items():
    #         tran = 'ε' if tran == '' else tran
    #         G.add_node(pydot.Node(start, shape='circle', style='bold' if start in self.finals else ''))
    #         for end in destinations:
    #             G.add_node(pydot.Node(end, shape='circle', style='bold' if end in self.finals else ''))
    #             G.add_edge(pydot.Edge(start, end, label=tran, labeldistance=2))

    #     G.add_edge(pydot.Edge('start', self.start, label='', style='dashed'))
    #     return G

    def _repr_svg_(self):
        try:
            return self.graph().create_svg().decode('utf8')
        except:
            pass

class DFA(NFA):
    
    def __init__(self, states, finals, transitions, start=0):
        assert all(isinstance(value, int) for value in transitions.values())
        assert all(len(symbol) > 0 for origin, symbol in transitions)
        
        transitions = { key: [value] for key, value in transitions.items() }
        NFA.__init__(self, states, finals, transitions, start)
        self.current = start
        
    def _move(self, symbol):
        if self.current in self.transitions:
            if symbol in self.transitions[self.current]:
                return self.transitions[self.current][symbol][0]
        return None
        
    
    def _reset(self):
        self.current = self.start
        
    def recognize(self, string):
        self._reset()
        for symbol in string:
            self.current = self._move(symbol)
        return self.current in self.finals
    
def move(automaton, states, symbol):
    moves = set()
    for state in states:
        if (not symbol in automaton.transitions[state]):
            continue  
        for i in automaton.transitions[state][symbol]:
            moves.add(i)  
    return moves
    
def epsilon_closure(automaton, states):
    pending = [ s for s in states ] # equivalente a list(states) pero me gusta así :p
    closure = { s for s in states } # equivalente a  set(states) pero me gusta así :p
    
    while pending:
        state = pending.pop()
        if (state,'') in automaton.map:
            for x in automaton.map[(state,'')]:
                closure.add(x)
                pending.append(x)
                
    return ContainerSet(*closure)

def nfa_to_dfa(automaton: NFA):
    transitions = {}
    
    start = epsilon_closure(automaton, [automaton.start])
    start.id = 0
    start.is_final = any(s in automaton.finals for s in start)
    states = [ start ]
    pending = [ start ]
    current_id = 1
    while pending:
        state = pending.pop()
        for symbol in automaton.vocabulary:
            # Your code here
            nfa_transitions_set = set()
                
            try:
                transitions[state.id, symbol]
                assert False, 'Invalid DFA!!!'
            except KeyError:
                # Your code here
                moves = move(automaton, list(state), symbol)
                new_state = epsilon_closure(automaton, list(moves))
                
                if len(new_state) > 0:
                    if new_state != state:
                        viewed_status = None
                        try:
                            viewed_status = states.index(new_state)
                        except ValueError:
                            pass

                        if viewed_status is None:
                            new_state.id = len(states) 
                            new_state.is_final = any(s in automaton.finals for s in new_state)
                            pending = [new_state] + pending
                            states.append(new_state)
                        else:
                            new_state.id = states[viewed_status].id
                            
                        transitions[state.id, symbol] = new_state.id
                    else :
                        transitions[state.id, symbol] = state.id
        
    finals = [ state.id for state in states if state.is_final ]
    dfa = DFA(len(states), finals, transitions)
    return dfa



def automata_union(a1, a2):
    
    states = a1.states + a2.states + 1
    start = 0
    transitions = {}

    # Estados finales
    finals1 = {c+1 for c in a1.finals}
    finals2 = {c+a1.states+1 for c in a2.finals}
    finals = set.union(finals1,finals2)

    for (origin,symb),dests in a1.map.items():
        if (origin,symb) in transitions:
            for dest in dests:
                transitions[origin+1,symb].append(dest+1)
        else:
            transitions[origin+1,symb] = [dest+1 for dest in dests]

    for (origin,symb),dests in a2.map.items():
        new_origin = origin+a1.states+1
        if (new_origin,symb) in transitions:
            for dest in dests:
                transitions[new_origin,symb].append(dest+a1.states+1)
        else:
            transitions[new_origin,symb] = [dest+a1.states+1 for dest in dests]
    
    # Agregar las transiciones del primer estado a los estados iniciales de los automatas
    transitions[0,''] = [1]
    transitions[0,''].append(a1.states+1)

    return NFA(states, finals, transitions, start)

def automata_concatenation(a1:NFA, a2:NFA):
    
    states = a1.states + a2.states
    start = a1.start
    finals = {a1.states+c for c in a2.finals}
    transitions = {}

    for (origin,symb),dests in a1.map.items():
        if (origin,symb) in transitions:
            for dest in dests:
                transitions[origin,symb].append(dest)
        else:
            transitions[origin,symb] = [dest for dest in dests]
    
    
    for (origin,symb),dests in a2.map.items():
        new_origin = origin+a1.states
        if (new_origin,symb) in transitions:
            for dest in dests:
                transitions[new_origin,symb].append(dest+a1.states)
        else:
            transitions[new_origin,symb] = [dest+a1.states for dest in dests]
    
    for f in a1.finals:
        if (f,'') in transitions:
            transitions[f,''].append(a1.states)
        else:
            transitions[f,''] = [a1.states]
    
    return NFA(states, finals, transitions, start)

def automata_closure(a1:NFA):  # Funciona
    
    states = a1.states
    start = a1.start
    finals = a1.finals
    transitions = {}
    
    for (origin,symb),dests in a1.map.items():
        transitions[(origin,symb)] = [dest for dest in dests]
    
    for state in finals:
        transitions[state,''] = [start]

    finals.add(start)

    
    return NFA(states, finals, transitions, start)



class Node:
    def evaluate(self):
        raise NotImplementedError()
        
class AtomicNode(Node):
    def __init__(self, lex):
        self.lex = lex

class UnaryNode(Node):
    def __init__(self, node):
        self.node = node
        
    def evaluate(self):
        value = self.node.evaluate() 
        return self.operate(value)
    
    @staticmethod
    def operate(value):
        raise NotImplementedError()
        
class BinaryNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def evaluate(self):
        lvalue = self.left.evaluate() 
        rvalue = self.right.evaluate()
        return self.operate(lvalue, rvalue)
    
    @staticmethod
    def operate(lvalue, rvalue):
        raise NotImplementedError()
    


EPSILON = 'ε'
class EpsilonNode(AtomicNode):
    def evaluate(self):
        nfa = NFA(states=2, finals=[1], transitions={
            (0,'ε'):[1]
        })    
        return nfa

class SymbolNode(AtomicNode):
    def evaluate(self):
        s = self.lex
        nfa = NFA(states=2, finals=[1], transitions={
            (0,s):[1]
        })
        return nfa
       
class ClosureNode(UnaryNode):
    @staticmethod
    def operate(value : NFA):
        nfa = automata_closure(value)
        return nfa
    
class UnionNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        nfa = automata_union(lvalue,rvalue)
        return nfa

class ConcatNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        nfa = automata_concatenation(lvalue,rvalue)
        return nfa