from cmp.automata import State
from cmp.tools.regex import Regex
from cmp.utils import Token

class Lexer:
    def __init__(self, table, eof):
        self.eof = eof
        self.regexs = self._build_regexs(table)
        self.automaton = self._build_automaton()
    
    def _build_regexs(self, table):
        regexs = []
        for _, (_, regex) in enumerate(table):
            autom = State.from_nfa(Regex(regex).automaton)
            regexs.append(autom)
        return regexs
    
    def _build_automaton(self):
        start = State('start')
        
        for regex in self.regexs:
            start.add_epsilon_transition(regex)
            
        return start.to_deterministic()
    
        
    def _walk(self, string):
        state = self.automaton
        final = state if state.final else None
        final_lex = lex = ''
        
        for symbol in string:
            new_state = state[symbol][0]
            
            final_lex = final_lex + symbol
            
            if new_state is None:
                if state.final:
                    return state,final_lex
                else:
                    return None,lex
            
            state = new_state
            lex = final_lex
            
        return final, final_lex
    
    def _tokenize(self, text):
        
        tmp = text
        
        while text:
            final, final_lex = self._walk(text)
            yield final_lex,final
            
            text = text.lstrip(final_lex)
        
        yield '$', self.eof
    
    def __call__(self, text):
        return [ Token(lex, ttype) for lex, ttype in self._tokenize(text) ]
    

nonzero_digits = '|'.join(str(n) for n in range(1,10))
letters = '|'.join(chr(n) for n in range(ord('a'),ord('z')+1))

print('Non-zero digits:', nonzero_digits)
print('Letters:', letters)

lexer = Lexer([
    ('num', f'({nonzero_digits})(0|{nonzero_digits})*'),
    ('for' , 'for'),
    ('foreach' , 'foreach'),
    ('space', '  *'),
    ('id', f'({letters})({letters}|0|{nonzero_digits})*')
], 'eof')

text = '5465 for 45foreach fore'
print(f'\n>>> Tokenizando: "{text}"')
tokens = lexer(text)
print(tokens)

