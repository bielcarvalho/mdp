O arquivo tem:

1. Os estados separados por vírgulas. 

2. As transições para cada ação definidas da seguinte forma:

action nome_da_ação
	estado_corrente estado_sucessor probabilidade_da_ação descartar
end_action

3. O custo de cada par estado ação.

4. O estado inicial.

5. O estado meta.

6. Um grid que é apenas para visualização. No grid:

1 - Parede
2 - Estado Inicial
3 - Estado Final
4 - Marcação que indica que há uma parede do lado



