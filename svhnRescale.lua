require 'image'

img_size = 28
D = torch.load('./housenumbers/train_32x32.t7','ascii')
Y = torch.Tensor(D.X:size(1),img_size,img_size)

print(Y:size())
for img = 1,Y:size(1) do
   xlua.progress(img,Y:size(1))
   temp = image.scale(D.X[{img}],img_size,img_size):float():div(255)
   Y[{img}] = 0.21 * temp[{1}] + 0.72 * temp[{1}] + 0.07 * temp[{1}]
   --A = (0.21 * temp[{1}] + 0.72 * temp[{2}] + 0.07 * temp[{3}])
   --print(A)
   --image.display(A)
   --image.display(temp)
end
D2 = {X=Y,y=D.y}

torch.save('./housenumbers/train_28x28.t7',D2)
