%word_topic=csvread('word-topic.csv');
%vocab=readtable('vocab.csv');
for i=1:20
    [val1,index1]=sort(word_topic(:, i),'descend');
    for j=1:5
        c(i,j)=vocab(index1(j),1);
    end
end