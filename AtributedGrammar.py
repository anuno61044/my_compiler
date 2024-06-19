
def evaluate_parse(left_parse, tokens):
    
    if not left_parse or not tokens:
        return
    
    left_parse = iter(left_parse)
    tokens = iter(tokens)
    next(tokens)
    result = evaluate(next(left_parse), left_parse, tokens)
    
    return result
    
    
def evaluate(production, left_parse, tokens, inherited_value=None):
    _, body = production
    
    attributes = production.attributes
    synteticed = []
    inherited = []
    for i in range(len(attributes)):
        synteticed.append(None)
        inherited.append(None)
    
    inherited[0] = inherited_value
    
    for i, symbol in enumerate(reversed(body),1):
        index = len(body)-i
        if symbol.IsTerminal:
            a = next(tokens).lex
            synteticed[index+1] = a
        else:
            next_production = next(left_parse)
            synteticed[index+1] = evaluate(next_production,left_parse,tokens, inherited_value)

    return attributes[0](inherited, synteticed)
    
