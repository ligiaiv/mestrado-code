# import heapq
# def ClosestXdestinations(numDestinations, allLocations, numDeliveries):
    
#     distances = [(coord[0]**2 + coord[1]**2) for coord in allLocations]
#     print(distances)
#     a = list(zip(*heapq.nsmallest(2, enumerate(distances))))[0]


#     # a = heapq.nlargest(2, enumerate(distances))
#     # heapq.n
#     print(a)

import heapq
def ClosestXdestinations(numDestinations, allLocations, numDeliveries):
    
    distances = [coord[0]**2 + coord[1]**2 for coord in allLocations]
    print(distances)
    closest = list(zip(*heapq.nsmallest(numDeliveries, enumerate(distances),key=lambda x:x[1])))[0]
    print("asd",heapq.nsmallest(numDeliveries, enumerate(distances), key=lambda x:x[1]))
    result = [tuple(allLocations[x]) for x in closest]
    
    # WRITE YOUR CODE HERE
    return result

INPUT = [[1,-3],[1,2],[3,4],[1,2]]
INPUT2 = [[3,6],[2,4],[5,3],[2,7],[1,8],[7,9]]
result = ClosestXdestinations(len(INPUT),INPUT,3)
print(result)