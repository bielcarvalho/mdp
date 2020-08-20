# Processos de decisão markovianos

O projeto foi desenvolvido em Python 3.7, com as bibliotecas SciPy e NumPy, na disciplina de Tópicos em Planejamento em Inteligência Artificial do curso de Sistemas de Informação da Escola de Artes, Ciências e Humanidades da Universidade de São Paulo. Ele teve como intuito comparar o desempenho dos algoritmos de Iteração de Valor e de Política em MDPs estocásticos de caminho mínimo.
A representação das probabilidades de transição foi feita por meio de matrizes esparsas, sendo mais adequada para problemas com um número menor de transições por ação. <br>
O arquivo mdp.py contém os algoritmos de iteração de valor e de política, enquanto base.py realiza a leitura dos arquivos de entrada e o armazenamento dos resultados.
O modelo dos arquivos de entrada podem ser vistos na pasta test_cases.

## Execução
A execução de base.py aceita os seguintes parâmetros de entrada:
-	-i *path* ou --input_path *path*: define o caminho da pasta ou arquvio de entrada;
-	-o *path* ou --output_path *path*: define o caminho da pasta de saída (opcional);
-	-gw ou --grid_world: usar para representar a política gerada por meio de um grid;
-	-vi ou --value_iteration: executar a iteração de valor para os arquivos na pasta de entrada;
-	-pi ou --policy_iteration: executar a iteração de política para os arquivos na pasta de entrada;
-	-mpi ou --modified_policy_iteration: executar a iteração de política modificada para os arquivos na pasta de entrada;
-	-e num ou --epsilon num: épsilon a ser utilizado (padrão é 0,1).

## Referências

* CORDWELL, Steven A W; GONZALEZ, Yasser; TULABANDHULA, Theja. Markov Decision Process (MDP) Toolbox for Python. [S. l.], 13 abr. 2015. Disponível em: https://github.com/sawcordwell/pymdptoolbox/.
* MAUSAM; KOLOBOV, Andrey. Planning with Markov Decision Processes: An AI Perspective. [S. l.]: Morgan & Claypool Publishers, 2012. ISBN 9781608458875.
