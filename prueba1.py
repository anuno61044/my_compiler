# import pydot
# from cmp.utils import ContainerSet

# class NFA:
#     def __init__(self, states, finals, transitions, start=0):
#         self.states = states
#         self.start = start
#         self.finals = set(finals)
#         self.map = transitions
#         self.vocabulary = set()
#         self.transitions = { state: {} for state in range(states) }
        
#         for (origin, symbol), destinations in transitions.items():
#             assert hasattr(destinations, '__iter__'), 'Invalid collection of states'
#             self.transitions[origin][symbol] = destinations
#             self.vocabulary.add(symbol)
            
#         self.vocabulary.discard('')
        
#     def epsilon_transitions(self, state):
#         assert state in self.transitions, 'Invalid state'
#         try:
#             return self.transitions[state]['']
#         except KeyError:
#             return ()
            
#     def graph(self):
#         G = pydot.Dot(rankdir='LR', margin=0.1)
#         G.add_node(pydot.Node('start', shape='plaintext', label='', width=0, height=0))

#         for (start, tran), destinations in self.map.items():
#             tran = 'ε' if tran == '' else tran
#             G.add_node(pydot.Node(start, shape='circle', style='bold' if start in self.finals else ''))
#             for end in destinations:
#                 G.add_node(pydot.Node(end, shape='circle', style='bold' if end in self.finals else ''))
#                 G.add_edge(pydot.Edge(start, end, label=tran, labeldistance=2))

#         G.add_edge(pydot.Edge('start', self.start, label='', style='dashed'))
#         return G

#     def _repr_svg_(self):
#         try:
#             return self.graph().create_svg().decode('utf8')
#         except:
#             pass
          
# class DFA(NFA):
    
#     def __init__(self, states, finals, transitions, start=0):
#         assert all(isinstance(value, int) for value in transitions.values())
#         assert all(len(symbol) > 0 for origin, symbol in transitions)
        
#         transitions = { key: [value] for key, value in transitions.items() }
#         NFA.__init__(self, states, finals, transitions, start)
#         self.current = start
        
#     def _move(self, symbol):
#         if self.current in self.transitions:
#             if symbol in self.transitions[self.current]:
               
#                 return self.transitions[self.current][symbol][0]
#         return None
        
    
#     def _reset(self):
#         self.current = self.start
        
#     def recognize(self, string):
#         self._reset()
#         for symbol in string:
#             self.current = self._move(symbol)
#         return self.current in self.finals

# def move(automaton, states, symbol):
#     moves = set()
#     for state in states:
#         if (not symbol in automaton.transitions[state]):
#             continue  
#         for i in automaton.transitions[state][symbol]:
#             moves.add(i)  
#     return moves

# def epsilon_closure(automaton, states):
#     pending = [ s for s in states ] # equivalente a list(states) pero me gusta así :p
#     closure = { s for s in states } # equivalente a  set(states) pero me gusta así :p
    
#     while pending:
#         state = pending.pop()
#         if ('' in automaton.transitions[state]):
#             for x in automaton.transitions[state]['']:
#                 closure.add(x)
#                 pending.append(x)
                
#     return ContainerSet(*closure)

# def nfa_to_dfa(automaton: NFA):
#     transitions = {}
    
#     start = epsilon_closure(automaton, [automaton.start])
#     start.id = 0
#     start.is_final = any(s in automaton.finals for s in start)
#     states = [ start ]
#     pending = [ start ]
#     current_id = 1
#     while pending:
#         state = pending.pop()
#         for symbol in automaton.vocabulary:
#             # Your code here
#             nfa_transitions_set = set()
                
#             try:
#                 transitions[state.id, symbol]
#                 assert False, 'Invalid DFA!!!'
#             except KeyError:
#                 # Your code here
#                 moves = move(automaton, list(state), symbol)
#                 new_state = epsilon_closure(automaton, list(moves))
                
#                 if len(new_state) > 0:
#                     if new_state != state:
#                         viewed_status = None
#                         try:
#                             viewed_status = states.index(new_state)
#                         except ValueError:
#                             pass

#                         if viewed_status is None:
#                             new_state.id = len(states) 
#                             new_state.is_final = any(s in automaton.finals for s in new_state)
#                             pending = [new_state] + pending
#                             states.append(new_state)
#                         else:
#                             new_state.id = states[viewed_status].id
                            
#                         transitions[state.id, symbol] = new_state.id
#                     else :
#                         transitions[state.id, symbol] = state.id
        
#     finals = [ state.id for state in states if state.is_final ]
#     dfa = DFA(len(states), finals, transitions)
#     return dfa


# automaton = NFA(states=6, finals=[3, 5], transitions={
#     (0, ''): [ 1, 2 ],
#     (1, ''): [ 3 ],
#     (1,'b'): [ 4 ],
#     (2,'a'): [ 4 ],
#     (3,'c'): [ 3 ],
#     (4, ''): [ 5 ],
#     (5,'d'): [ 5 ]
# })


from cmp.pycompiler import Grammar
from TreeRegex import *

G = Grammar()

E = G.NonTerminal('E', True)
T, F = G.NonTerminals('T F')
pipe, star, opar, cpar, symbol = G.Terminals('| * ( ) symbol')

# > PRODUCTIONS??? LR(1) Grammar
E %= E + pipe + T, lambda h,s: UnionNode(s[1],s[3])
E %= T, lambda h,s: s[1]

T %= T + F, lambda h,s: ConcatNode(s[1],s[2])
T %= F, lambda h,s: s[1]

F %= opar + E + cpar, lambda h,s: s[2]
F %= F + star, lambda h,s: ClosureNode(s[1])
F %= symbol, lambda h,s: SymbolNode(s[1])



from cmp.utils import Token

def regex_tokenizer(text, G, skip_whitespaces=True):
    tokens = []
    fixed_tokens = ['|']
    tmp = ''
    text = text + '$'
    for char in text:
        if skip_whitespaces and char.isspace():
            continue
        if char == '|':
            if len(tmp) > 0:
                tokens.append(Token(tmp,symbol))
            tokens.append(Token(char,pipe))
            tmp = ''
        elif char == '*':
            if len(tmp) > 0:
                tokens.append(Token(tmp,symbol))
            tokens.append(Token(char,star))
            tmp = ''
        elif char == '(':
            if len(tmp) > 0:
                tokens.append(Token(tmp,symbol))
            tokens.append(Token(char,opar))
            tmp = ''
        elif char == ')':
            if len(tmp) > 0:
                tokens.append(Token(tmp,symbol))
            tokens.append(Token(char,cpar))
            tmp = ''
        elif char == '$':
            if len(tmp) > 0:
                tokens.append(Token(tmp,symbol))
            tokens.append(Token('$', G.EOF))
            break
        else:
            tmp = tmp + char
        
    return tokens

from cmp.ast import get_printer


tokens = regex_tokenizer('(a*|b*)*b',G)

from parser_tools1 import *

parser = LR1Parser(G)
print([tok.token_type for tok in tokens])

derivations = parser([tok.token_type for tok in tokens])
print(derivations)

tokens.reverse()
derivations.reverse()

result = evaluate_parse(derivations, tokens)

printer = get_printer(AtomicNode=SymbolNode, BinaryNode=BinaryNode, UnaryNode=ClosureNode)
print(printer(result))


nfa = result.evaluate()

# dfa = nfa_to_dfa(nfa)

# print(dfa.recognize('aaaaab'))
