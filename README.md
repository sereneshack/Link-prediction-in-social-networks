# Link-prediction-in-social-networks

We have picked Link Prediction in social networks as our project. Currently, with
the rapid development, the online social network has been a part of people’s life.
A lot of sociology, biology, and information systems can use the network to
describe, in which nodes represent individual and edges represent the
relationships between individuals or the interaction between individuals.

Social networks are social structures including some actors and relationships
amid them. These networks are presented by employing some nodes and ties. The
ties show some type of relationships among the nodes including kinships,
friendships, collaborations, and any other interactions between the people in the
network. Link prediction is an important research field in data mining. It has a
wide range of scenarios. Many data mining tasks involve the relationship between
the objects. Link prediction can be used for recommendation systems, social
networks, information retrieval, and many other fields.
Given a graph G={V, E} of the social network at a moment of the node and the
other node, link prediction is to predict the probability of the link between the
node and the other node.
The two challenges we will be facing are:
1) Predict that the new link will appear in future time.
2) Forecast hidden unknown link in the space.
In our project, we have combined all these aspects mentioned above in a
single problem statement using different algorithms SVM(Support Vector
Machine), ANN, Logistic regression, Fuzzy model, PSO, ACO, hybrid ACO & PSO.
In the end, we are comparing the accuracy of all these algorithms on different
datasets of different social media platforms.

The link prediction problem is usually described as:
Given a set of data instances V = v in i=1 ,
which is organized in the form of a social network
G = (V,E)
where E is the set of observed links
Then the task to predict how likely an unobserved link e ij ∉ E exists between an
arbitrary pair of nodes v i , v j in the data network.
Given a snapshot of a social network at time t (or network evolution between (t 1
and t 2 ), seek to accurately predict the edges that will be added to the network
during the interval from time t to a given future time t′. The easiest framework of link prediction algorithm is based on the similarity of the
algorithm. Any pair of node and node,we have assigned to this node is a
function,Similarly this function is defined as the similarity function between nodes
and . Then sorting the nodes pair in accordance with the function values from the
largest to smallest, the greater the value of the similarity function, the greater the
probability of the link in the nodes.

Dataset: http://snap.stanford.edu/data/
