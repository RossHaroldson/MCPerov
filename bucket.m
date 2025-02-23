function bin = bucket(dist,edges)
bin=0;
for i = 1:length(dist)
    if (dist(i) - edges(1)) > 0 && (dist(i) - edges(2)) <= 0
        bin = bin + 1;
    end
    if dist(i) == 0 && edges(1)==0
        bin = bin + 1;
    end
end
end