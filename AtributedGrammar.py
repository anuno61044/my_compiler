def evaluate_parse(left_parse, tokens):
    if not left_parse or not tokens:
        return
    
    left_parse = iter(left_parse)
    tokens = iter(tokens)
    result = evaluate(next(left_parse), left_parse, tokens)
    
    return result
    

def evaluate(production, left_parse, tokens, inherited_value=None):
    head, body = production
    attributes = production.attributes
    
    # Insert your code here ...
    # > synteticed = ...
    # > inherited = ...
    synteticed = []
    inherited = []
    for i in range(len(attributes)):
        synteticed.append(None)
        inherited.append(None)
    
    inherited[0] = inherited_value
    for i, symbol in enumerate(body, 1):
        if symbol.IsTerminal:
            assert inherited[i] is None
            if(body[i - 1] == num):
                synteticed[i] = next(tokens).lex
                inherited_value = attributes[0](inherited,synteticed)
                return inherited_value
            else:
                synteticed[i] = next(tokens).lex
                # attributes[i](inherited,synteticed)
        else:
            next_production = next(left_parse)
            assert symbol == next_production.Left
            if not attributes[i] == None:
               synteticed[i] = evaluate(next_production, left_parse, tokens, attributes[i](inherited, synteticed))
            else :
                synteticed[i] = evaluate(next_production,left_parse,tokens, inherited_value)
    # Insert your code here ...
    return attributes[0](inherited, synteticed)
    
