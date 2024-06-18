from cmp.utils import pprint, inspect
from cmp.pycompiler import *
from TreeDef import *

G = Grammar()

# No terminales
E = G.NonTerminal('E', True)
T,F = G.NonTerminals('T F')

# Terminales
equal, plus, times, num, lpar, rpar = G.Terminals('= + * float ( )')

# Reglas
# E %= A, lambda h,s : s[1]

E %=  E + plus + T, lambda h,s: PlusNode(s[1],s[3])
E %= T, lambda h,s : s[1]

T %= T + times + F, lambda h,s: StarNode(s[1],s[3])
T %= F, lambda h,s : s[1]

F %= num, lambda h,s : ConstantNumberNode(s[1])
F %= lpar + E + rpar, lambda h,s : s[2]



from cmp.pycompiler import Item

# item = Item(E.productions[0], 0, lookaheads=[G.EOF, plus])
# print('item:', item)


from cmp.utils import ContainerSet
from cmath import e


def compute_local_first(firsts, alpha):
    first_alpha = ContainerSet()
    
    try:
        alpha_is_epsilon = alpha.IsEpsilon
    except:
        alpha_is_epsilon = False
    
    ###################################################
    # alpha == epsilon ? First(alpha) = { epsilon }
    ###################################################
    if(alpha_is_epsilon):
        first_alpha.set_epsilon(True)
        return first_alpha
    ###################################################
    
    ###################################################
    # alpha = X1 ... XN
    # First(Xi) subconjunto First(alpha)
    # epsilon pertenece a First(X1)...First(Xi) ? First(Xi+1) subconjunto de First(X) y First(alpha)
    # epsilon pertenece a First(X1)...First(XN) ? epsilon pertence a First(X) y al First(alpha)
    ###################################################
    for symbol in alpha:
        first_symbol = firsts[symbol]
        first_alpha.update(first_symbol)
        if(not first_symbol.contains_epsilon):
            break
            
    ###################################################
    
    # First(alpha)
    return first_alpha

def compute_firsts(G):
    firsts = {}
    change = True
    
    # init First(Vt)
    # Each terminal has a First set with only itself
    for terminal in G.terminals:
        firsts[terminal] = ContainerSet(terminal)
    
    # init First(Vn)
    # Each non-terminal is initialized with an empty set
    for nonterminal in G.nonTerminals:
        firsts[nonterminal] = ContainerSet()
    
    while change:
        change = False
        
        # P: X -> alpha
        for production in G.Productions:
            X = production.Left
            alpha = production.Right
            
            # get current First(X)
            first_X = firsts[X]
                
            # init First(alpha)
            try:
                first_alpha = firsts[alpha]
            except KeyError:
                first_alpha = firsts[alpha] = ContainerSet()
            
            # CurrentFirst(alpha)???
            local_first = compute_local_first(firsts, alpha)
            # update First(X) and First(alpha) from CurrentFirst(alpha)
            change |= first_alpha.hard_update(local_first)
            change |= first_X.hard_update(local_first)
                    
    return firsts

def expand(item: Item, firsts):
    next_symbol = item.NextSymbol
    if next_symbol is None or not next_symbol.IsNonTerminal:
        return []
    
    lookaheads = ContainerSet()

    # Your code here!!! (Compute lookahead for child items)
    for x in item.Preview():
        lookaheads.update(compute_local_first(firsts, x))
        
    # assert not lookaheads.contains_epsilon
    productions = next_symbol.productions
    # Your code here!!! (Build and return child items)
    return [Item(production, 0, lookaheads) for production in productions]
    
def compress(items):
    centers = {}

    for item in items:
        center = item.Center()
        try:
            lookaheads = centers[center]
        except KeyError:
            centers[center] = lookaheads = set()
        lookaheads.update(item.lookaheads)
    
    return { Item(x.production, x.pos, set(lookahead)) for x, lookahead in centers.items() }

def closure_lr1(items, firsts):
    closure = ContainerSet(*items)
    
    changed = True
    while changed:
        changed = False
        
        new_items = ContainerSet()
        
        # Your code here!!!
        for x in closure:
            expanded=expand(x, firsts)
            new_items.update(ContainerSet(*expanded))
                    
        changed = closure.update(new_items)
        
    return compress(closure)
   
def goto_lr1(items, symbol, firsts=None, just_kernel=False):
    assert just_kernel or firsts is not None, '`firsts` must be provided if `just_kernel=False`'
    items = frozenset(item.NextItem() for item in items if item.NextSymbol == symbol)
    return items if just_kernel else closure_lr1(items, firsts)

    
from pprint import pprint
from cmp.automata import State, multiline_formatter

def build_LR1_automaton(G):
    # assert len(G.startSymbol.productions) == 1, 'Grammar must be augmented'
    
    firsts = compute_firsts(G)
    firsts[G.EOF] = ContainerSet(G.EOF)
    
    start_production = G.startSymbol.productions[0]
    start_item = Item(start_production, 0, lookaheads=(G.EOF,))
    start = frozenset([start_item])
    
    closure = closure_lr1(start, firsts)
    # print('closure:', [type(x) for x in closure])
    automaton = State(frozenset(closure), True)
    # print('automaton', automaton.state)
    pending = [ start ]
    visited = { start: automaton }
    # print('intento', closure_lr1(automaton.state, firsts))
    while pending:
        current = pending.pop()
        
        current_state = visited[current]
          
        new_closure = closure_lr1(current_state.state, firsts)
        for new_item in new_closure:
            for symbol in G.terminals + G.nonTerminals:
                next_state = None
                if symbol.Name in current_state.transitions:
                    continue
                if new_item.NextSymbol == symbol:
                    # print('current state and symbol', current_state, symbol)
                    next_state = goto_lr1(new_closure, symbol, firsts=firsts)
                    # print('next state', next_state)
                    
                    if next_state:
                        frozen = frozenset(next_state)
                        # print('el frozen', frozen)
                        if not frozen in visited:
                            # print('lo pongo')
                            pending.append(frozen)
                            new_state = State(frozen, True)
                            visited[frozen] = new_state
                            current_state.add_transition(symbol.Name, new_state)
                        else:
                            current_state.add_transition(symbol.Name, visited[frozen])
                    
            # if next_state:
    # print()
    for x in visited:
        state = visited[x]
        # print(state, state.transitions)
        print()
    print()
    automaton.set_formatter(multiline_formatter)
    return automaton

automaton = build_LR1_automaton(G.AugmentedGrammar())


class ShiftReduceParser:

    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'
    OK = 'OK'

    def __init__(self, G, verbose=False):
        self.G = G
        self.verbose = verbose
        self.action = {}
        self.goto = {}
        self._build_parsing_table()

    def _build_parsing_table(self):
        raise NotImplementedError()

    def __call__(self, w, get_shift_reduce=False):
        stack = [0]
        cursor = 0
        output = []
        operations = []
        while True:
            state = stack[-1]
            lookahead = w[cursor]

            if(state, lookahead) not in self.action:
                excepted_char = ''

                for (state1, i) in self.action.keys():
                    if i.IsTerminal and state1 == state:
                        excepted_char += str(i) + ', '
                parsed = ' '.join([str(m)
                                    for m in stack if not str(m).isnumeric()])
                message_error = f'It was expected "{excepted_char}" received "{lookahead}" after {parsed}'
                print("\nError. Aborting...")
                print('')
                print("\n", message_error)
                # print(w[cursor-1])
                return None

            if self.action[state, lookahead] == self.OK:
                action = self.OK
            else:
                action, tag = self.action[state, lookahead]
            # print('action, tsg', action)
            if action == self.SHIFT:
                operations.append(self.SHIFT)
                stack += [lookahead, tag]
                cursor += 1
            elif action == self.REDUCE:
                operations.append(self.REDUCE)
                output.append(tag)
                # print('tag', tag)
                head, body = tag
                for symbol in reversed(body):
                    # print('stack', stack)
                    stack.pop()

                    assert stack.pop() == symbol
                    state = stack[-1]
                    # print(self.goto,'goto')
                    # print('output', output)
                goto = self.goto[state, head]
                stack += [head, goto]
            elif action == self.OK:
                stack.pop()
                assert stack.pop() == self.G.startSymbol
                assert len(stack) == 1
                return output if not get_shift_reduce else(output, operations)
            else:
                raise Exception('Invalid action!!!')

class LR1Parser(ShiftReduceParser):
    def _build_parsing_table(self):
        G = self.G.AugmentedGrammar(True)
        
        automaton = build_LR1_automaton(G)
        for i, node in enumerate(automaton):
            if self.verbose: print(i, '\t', '\n\t '.join(str(x) for x in node.state), '\n')
            node.idx = i
        
        for node in automaton:
            idx = node.idx
            for item in node.state:
                print('current item', item)
                # Your code here!!!
                # - Fill self.Action and self.Goto according to item)
                
                    
                if  item.NextSymbol and item.NextSymbol.IsTerminal:
                    self._register(self.action, (idx, item.NextSymbol), (self.SHIFT,node.get(item.NextSymbol.Name).idx))
                    # self.action[idx, item.NextSymbol] = self.SHIFT,node.get(item.NextSymbol.Name).idx
                elif not item.NextSymbol and not item.production.Left == G.startSymbol:
                    
                    for lookahead in item.lookaheads:
                        self._register(self.action, (idx, lookahead), (self.REDUCE, item.production))
                        # self.action[idx, lookahead] = self.REDUCE, item.production
                
                elif item.IsReduceItem and item.production.Left == G.startSymbol and not item.NextSymbol:
                    
                    self._register(self.action, (idx, G.EOF), self.OK)

                else: #item.NextSymbol and item.NextSymbol.IsNonTerminal:
                    self._register(self.goto, (idx, item.NextSymbol), node.get(item.NextSymbol.Name).idx)
                # - Feel free to use self._register(...))
     
        
    @staticmethod
    def _register(table, key, value):
        assert key not in table or table[key] == value, 'Shift-Reduce or Reduce-Reduce conflict!!!'
        table[key] = value
        
parser = LR1Parser(G, verbose=True)

def encode_value(value):
    try:
        action, tag = value
        if action == ShiftReduceParser.SHIFT:
            return 'S' + str(tag)
        elif action == ShiftReduceParser.REDUCE:
            return repr(tag)
        elif action ==  ShiftReduceParser.OK:
            return action
        else:
            return value
    except TypeError:
        return value
    

class Token:
    """
    Basic token class. 
    
    Parameters
    ----------
    lex : str
        Token's lexeme.
    token_type : Enum
        Token's type.
    """
    
    def __init__(self, lex, token_type):
        self.lex = lex
        self.token_type = token_type
    
    def __str__(self):
        return f'{self.token_type}: {self.lex}'
    
    def __repr__(self):
        return str(self)
    

def evaluate_parse(left_parse, tokens):
    
    if not left_parse or not tokens:
        return
    
    left_parse = iter(left_parse)
    tokens = iter(tokens)
    next(tokens)
    result = evaluate(next(left_parse), left_parse, tokens)
    
    # assert isinstance(next(tokens).token_type, EOF)
    return result
    
def evaluate(production, left_parse, tokens, inherited_value=None):
    head, body = production
    # print('head ',head)
    # print('body ',body)
    
    print('nueva evaluacion : ', production)
    attributes = production.attributes
    # print('atributes ', attributes)
    synteticed = []
    inherited = []
    for i in range(len(attributes)):
        synteticed.append(None)
        inherited.append(None)
    
    inherited[0] = inherited_value
    
    for i, symbol in enumerate(reversed(body),1):
        index = len(body)-i
        print('voy a analizar el simbolo ', symbol)
        if symbol.IsTerminal:
            print('reconoci un terminal')
            # print('body[i-1] ',body[i - 1])
            a = next(tokens).lex
            print('token a analizar y su posicion', a, index+1)
            synteticed[index+1] = a
        else:
            next_production = next(left_parse)
            
            synteticed[index+1] = evaluate(next_production,left_parse,tokens, inherited_value)
    # Insert your code here ...
    print('termine una evaluacion ', production, attributes[0](inherited, synteticed))
    print('sintetizados ', synteticed)
    print('\n')
    return attributes[0](inherited, synteticed)
    


tokens = [ Token('5', num), Token('*', times), Token('(', lpar), Token('2', num), Token('+', plus), Token('4', num), Token(')', rpar), Token('$', G.EOF) ]
derivation = parser([num, times, lpar, num, plus, num, rpar, G.EOF])

tokens.reverse()
derivation.reverse()

print('\n\n',derivation)
print(tokens)
print('\n')

result = evaluate_parse(derivation, tokens)
from cmp.ast import get_printer
printer = get_printer(AtomicNode=ConstantNumberNode, BinaryNode=BinaryNode)
print(printer(result))