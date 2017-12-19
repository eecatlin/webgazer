dimension=3
numberofdatapoints=15
numberofcentroids=3
numberofiterations=5
startinglocations=zeros(numberofcentroids,dimension,numberofiterations+1)
%startinglocations(:,:,1)=[2 2; 13 12; 35 32]
%starting locations matter, pick data points as starting locations
categorizationmatrix=zeros(numberofcentroids,numberofdatapoints,numberofiterations)
categorizedcoordinates=zeros(numberofdatapoints,dimension,numberofiterations,numberofcentroids)

data=zeros(15,2)
%data=[1 1; 3 1; 15 1; 7 3; 4 6; 3 3; 4 5; 5 4; 7 8; 8 2; 3 9; 7 15; 15 8; 6 2; 3 4; 12 7]
data=[1 1 1; 2 1 1; 1 2 1; 2 2 1; 0 1 1; 15 1 7; 15 2 8; 14 1 7; 14 2 7; 15 0 7; 15 15 15; 16 15 15; 14 15 14; 15 14 15; 14 14 15]
size(data)
stem3(data)
scatter3(data(:,1),data(:,2),data(:,3))
startinglocations(:,:,1)=data(1:numberofcentroids,1:dimension)
%startinglocations(:,:,1)=[15 15 15; 320 18 150; -52 -200 -300]
distancematrix=zeros(numberofcentroids,numberofdatapoints,numberofiterations)
for iterationnumber=1:numberofiterations
%iterationnumber=1

for n=1:numberofcentroids
for m=1:numberofdatapoints
    distancematrix(n,m,iterationnumber)=sqrt(sum((data(m,:)-startinglocations(n,:,iterationnumber)).^2))
end
end
stem3(distancematrix(:,:,iterationnumber))
[sorteddistancematrix,locations]=sort(distancematrix,1)
[nearestneighbormatrix, neighborindex]=sort(distancematrix,2)
stem3(sorteddistancematrix(:,:,iterationnumber))
for n=1:numberofcentroids
for m=1:numberofdatapoints
if n==locations(1,m,iterationnumber)
categorizationmatrix(n,m,iterationnumber)=locations(1,m,iterationnumber)
end

for q=1:numberofcentroids
    offthemapcheck=sum(categorizationmatrix,2)
    if offthemapcheck(q,1,iterationnumber)==0
        categorizationmatrix(:,neighborindex(q,1,iterationnumber),iterationnumber)=0
        categorizationmatrix(q,neighborindex(q,1,iterationnumber),iterationnumber)=q
    end
   
    
end
end

categorizationmatrix(:,:,1)
for n=1:numberofcentroids
for m=1:numberofdatapoints
    if n==categorizationmatrix(n,m,iterationnumber)
        categorizedcoordinates(m,:,iterationnumber,n)=data(m,:)
    end
end
end

categorizedcoordinates(:,:,1,1)
categorizedcoordinates(:,:,1,2)
categorizedcoordinates(:,:,1,3)
numberofpointsineachcategory=zeros(1,numberofcentroids,numberofiterations)
for n = 1:numberofcentroids
    numberofpointsineachcategory(:,n,iterationnumber)=sum(categorizationmatrix(n,:,iterationnumber))/max(categorizationmatrix(n,:,iterationnumber))
end
newcentroids=zeros(numberofcentroids,dimension)
for n=1:numberofcentroids
newcentroids(n,:)=sum(categorizedcoordinates(:,:,iterationnumber,n),1)/numberofpointsineachcategory(1,n,iterationnumber)
end
for n=1:numberofcentroids
startinglocations(n,:,iterationnumber+1)=newcentroids(n,:)
end
end
end
categorizationmatrix
offthemapcheck
startinglocations
