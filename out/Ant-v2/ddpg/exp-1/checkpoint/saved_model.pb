??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??

n
	pi/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name	pi/kernel
g
pi/kernel/Read/ReadVariableOpReadVariableOp	pi/kernel*
_output_shapes

:@*
dtype0
f
pi/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pi/bias
_
pi/bias/Read/ReadVariableOpReadVariableOppi/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:o@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:o@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
v
	_body
_pi
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer_with_weights-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
?
layer_regularization_losses
layer_metrics
	variables
regularization_losses

layers
metrics
trainable_variables
 non_trainable_variables
 
h

kernel
bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
q
%axis
	gamma
beta
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
y
.layer_with_weights-0
.layer-0
/	variables
0regularization_losses
1trainable_variables
2	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
3layer_regularization_losses
4layer_metrics
	variables
regularization_losses

5layers
6metrics
trainable_variables
7non_trainable_variables
DB
VARIABLE_VALUE	pi/kernel%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUE
@>
VARIABLE_VALUEpi/bias#_pi/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
8layer_regularization_losses
9layer_metrics
regularization_losses
	variables

:layers
;metrics
trainable_variables
<non_trainable_variables
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElayer_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
 
 

0
1

0
1
?
=layer_regularization_losses
>layer_metrics
!regularization_losses
"	variables

?layers
@metrics
#trainable_variables
Anon_trainable_variables
 
 

0
1

0
1
?
Blayer_regularization_losses
Clayer_metrics
&regularization_losses
'	variables

Dlayers
Emetrics
(trainable_variables
Fnon_trainable_variables
 
 
 
?
Glayer_regularization_losses
Hlayer_metrics
*regularization_losses
+	variables

Ilayers
Jmetrics
,trainable_variables
Knon_trainable_variables
h

kernel
bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api

0
1
 

0
1
?
Player_regularization_losses
Qlayer_metrics
/	variables
0regularization_losses

Rlayers
Smetrics
1trainable_variables
Tnon_trainable_variables
 
 

0
	1

2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1

0
1
?
Ulayer_regularization_losses
Vlayer_metrics
Lregularization_losses
M	variables

Wlayers
Xmetrics
Ntrainable_variables
Ynon_trainable_variables
 
 

.0
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????o*
dtype0*
shape:?????????o
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/bias	pi/kernelpi/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_46356229
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamepi/kernel/Read/ReadVariableOppi/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_46356954
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	pi/kernelpi/biasdense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_46356988??	
?N
?
C__inference_actor_layer_call_and_return_conditional_losses_46356709

inputs6
$dense_matmul_readvariableop_resource:o@3
%dense_biasadd_readvariableop_resource:@?
1layer_normalization_mul_2_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@@
2sequential_dense_1_biasadd_readvariableop_resource:@
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd|
layer_normalization/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshapedense/BiasAdd:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape?
layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
layer_normalization/ones/Less/y?
layer_normalization/ones/LessLesslayer_normalization/mul:z:0(layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2
layer_normalization/ones/Less?
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/ones/packed?
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
layer_normalization/ones/Const?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/ones?
 layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 layer_normalization/zeros/Less/y?
layer_normalization/zeros/LessLesslayer_normalization/mul:z:0)layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
layer_normalization/zeros/Less?
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2"
 layer_normalization/zeros/packed?
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer_normalization/zeros/Const?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/zerosy
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const}
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_1?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/addy
activation/TanhTanhlayer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
activation/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulactivation/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/Relu?
IdentityIdentity%sequential/dense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?~
?	
#__inference__wrapped_model_46355666
input_1O
=private__mlp_actor_actor_dense_matmul_readvariableop_resource:o@L
>private__mlp_actor_actor_dense_biasadd_readvariableop_resource:@X
Jprivate__mlp_actor_actor_layer_normalization_mul_2_readvariableop_resource:@V
Hprivate__mlp_actor_actor_layer_normalization_add_readvariableop_resource:@\
Jprivate__mlp_actor_actor_sequential_dense_1_matmul_readvariableop_resource:@@Y
Kprivate__mlp_actor_actor_sequential_dense_1_biasadd_readvariableop_resource:@F
4private__mlp_actor_pi_matmul_readvariableop_resource:@C
5private__mlp_actor_pi_biasadd_readvariableop_resource:
identity??5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp?4private__mlp_actor/actor/dense/MatMul/ReadVariableOp??private__mlp_actor/actor/layer_normalization/add/ReadVariableOp?Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp?Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp?Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp?,private__mlp_actor/pi/BiasAdd/ReadVariableOp?+private__mlp_actor/pi/MatMul/ReadVariableOp?
4private__mlp_actor/actor/dense/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_actor_dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype026
4private__mlp_actor/actor/dense/MatMul/ReadVariableOp?
%private__mlp_actor/actor/dense/MatMulMatMulinput_1<private__mlp_actor/actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%private__mlp_actor/actor/dense/MatMul?
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOpReadVariableOp>private__mlp_actor_actor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp?
&private__mlp_actor/actor/dense/BiasAddBiasAdd/private__mlp_actor/actor/dense/MatMul:product:0=private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
&private__mlp_actor/actor/dense/BiasAdd?
2private__mlp_actor/actor/layer_normalization/ShapeShape/private__mlp_actor/actor/dense/BiasAdd:output:0*
T0*
_output_shapes
:24
2private__mlp_actor/actor/layer_normalization/Shape?
@private__mlp_actor/actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@private__mlp_actor/actor/layer_normalization/strided_slice/stack?
Bprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_1?
Bprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_2?
:private__mlp_actor/actor/layer_normalization/strided_sliceStridedSlice;private__mlp_actor/actor/layer_normalization/Shape:output:0Iprivate__mlp_actor/actor/layer_normalization/strided_slice/stack:output:0Kprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_1:output:0Kprivate__mlp_actor/actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:private__mlp_actor/actor/layer_normalization/strided_slice?
2private__mlp_actor/actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :24
2private__mlp_actor/actor/layer_normalization/mul/x?
0private__mlp_actor/actor/layer_normalization/mulMul;private__mlp_actor/actor/layer_normalization/mul/x:output:0Cprivate__mlp_actor/actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 22
0private__mlp_actor/actor/layer_normalization/mul?
Bprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack?
Dprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_1?
Dprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_2?
<private__mlp_actor/actor/layer_normalization/strided_slice_1StridedSlice;private__mlp_actor/actor/layer_normalization/Shape:output:0Kprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack:output:0Mprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_1:output:0Mprivate__mlp_actor/actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<private__mlp_actor/actor/layer_normalization/strided_slice_1?
4private__mlp_actor/actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :26
4private__mlp_actor/actor/layer_normalization/mul_1/x?
2private__mlp_actor/actor/layer_normalization/mul_1Mul=private__mlp_actor/actor/layer_normalization/mul_1/x:output:0Eprivate__mlp_actor/actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 24
2private__mlp_actor/actor/layer_normalization/mul_1?
<private__mlp_actor/actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2>
<private__mlp_actor/actor/layer_normalization/Reshape/shape/0?
<private__mlp_actor/actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2>
<private__mlp_actor/actor/layer_normalization/Reshape/shape/3?
:private__mlp_actor/actor/layer_normalization/Reshape/shapePackEprivate__mlp_actor/actor/layer_normalization/Reshape/shape/0:output:04private__mlp_actor/actor/layer_normalization/mul:z:06private__mlp_actor/actor/layer_normalization/mul_1:z:0Eprivate__mlp_actor/actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:private__mlp_actor/actor/layer_normalization/Reshape/shape?
4private__mlp_actor/actor/layer_normalization/ReshapeReshape/private__mlp_actor/actor/dense/BiasAdd:output:0Cprivate__mlp_actor/actor/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????26
4private__mlp_actor/actor/layer_normalization/Reshape?
8private__mlp_actor/actor/layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2:
8private__mlp_actor/actor/layer_normalization/ones/Less/y?
6private__mlp_actor/actor/layer_normalization/ones/LessLess4private__mlp_actor/actor/layer_normalization/mul:z:0Aprivate__mlp_actor/actor/layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 28
6private__mlp_actor/actor/layer_normalization/ones/Less?
8private__mlp_actor/actor/layer_normalization/ones/packedPack4private__mlp_actor/actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2:
8private__mlp_actor/actor/layer_normalization/ones/packed?
7private__mlp_actor/actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7private__mlp_actor/actor/layer_normalization/ones/Const?
1private__mlp_actor/actor/layer_normalization/onesFillAprivate__mlp_actor/actor/layer_normalization/ones/packed:output:0@private__mlp_actor/actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????23
1private__mlp_actor/actor/layer_normalization/ones?
9private__mlp_actor/actor/layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2;
9private__mlp_actor/actor/layer_normalization/zeros/Less/y?
7private__mlp_actor/actor/layer_normalization/zeros/LessLess4private__mlp_actor/actor/layer_normalization/mul:z:0Bprivate__mlp_actor/actor/layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 29
7private__mlp_actor/actor/layer_normalization/zeros/Less?
9private__mlp_actor/actor/layer_normalization/zeros/packedPack4private__mlp_actor/actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2;
9private__mlp_actor/actor/layer_normalization/zeros/packed?
8private__mlp_actor/actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8private__mlp_actor/actor/layer_normalization/zeros/Const?
2private__mlp_actor/actor/layer_normalization/zerosFillBprivate__mlp_actor/actor/layer_normalization/zeros/packed:output:0Aprivate__mlp_actor/actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????24
2private__mlp_actor/actor/layer_normalization/zeros?
2private__mlp_actor/actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 24
2private__mlp_actor/actor/layer_normalization/Const?
4private__mlp_actor/actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 26
4private__mlp_actor/actor/layer_normalization/Const_1?
=private__mlp_actor/actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3=private__mlp_actor/actor/layer_normalization/Reshape:output:0:private__mlp_actor/actor/layer_normalization/ones:output:0;private__mlp_actor/actor/layer_normalization/zeros:output:0;private__mlp_actor/actor/layer_normalization/Const:output:0=private__mlp_actor/actor/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2?
=private__mlp_actor/actor/layer_normalization/FusedBatchNormV3?
6private__mlp_actor/actor/layer_normalization/Reshape_1ReshapeAprivate__mlp_actor/actor/layer_normalization/FusedBatchNormV3:y:0;private__mlp_actor/actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@28
6private__mlp_actor/actor/layer_normalization/Reshape_1?
Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOpReadVariableOpJprivate__mlp_actor_actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp?
2private__mlp_actor/actor/layer_normalization/mul_2Mul?private__mlp_actor/actor/layer_normalization/Reshape_1:output:0Iprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@24
2private__mlp_actor/actor/layer_normalization/mul_2?
?private__mlp_actor/actor/layer_normalization/add/ReadVariableOpReadVariableOpHprivate__mlp_actor_actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02A
?private__mlp_actor/actor/layer_normalization/add/ReadVariableOp?
0private__mlp_actor/actor/layer_normalization/addAddV26private__mlp_actor/actor/layer_normalization/mul_2:z:0Gprivate__mlp_actor/actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0private__mlp_actor/actor/layer_normalization/add?
(private__mlp_actor/actor/activation/TanhTanh4private__mlp_actor/actor/layer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2*
(private__mlp_actor/actor/activation/Tanh?
Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOpJprivate__mlp_actor_actor_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02C
Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp?
2private__mlp_actor/actor/sequential/dense_1/MatMulMatMul,private__mlp_actor/actor/activation/Tanh:y:0Iprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@24
2private__mlp_actor/actor/sequential/dense_1/MatMul?
Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpKprivate__mlp_actor_actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
3private__mlp_actor/actor/sequential/dense_1/BiasAddBiasAdd<private__mlp_actor/actor/sequential/dense_1/MatMul:product:0Jprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@25
3private__mlp_actor/actor/sequential/dense_1/BiasAdd?
0private__mlp_actor/actor/sequential/dense_1/ReluRelu<private__mlp_actor/actor/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@22
0private__mlp_actor/actor/sequential/dense_1/Relu?
+private__mlp_actor/pi/MatMul/ReadVariableOpReadVariableOp4private__mlp_actor_pi_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02-
+private__mlp_actor/pi/MatMul/ReadVariableOp?
private__mlp_actor/pi/MatMulMatMul>private__mlp_actor/actor/sequential/dense_1/Relu:activations:03private__mlp_actor/pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pi/MatMul?
,private__mlp_actor/pi/BiasAdd/ReadVariableOpReadVariableOp5private__mlp_actor_pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,private__mlp_actor/pi/BiasAdd/ReadVariableOp?
private__mlp_actor/pi/BiasAddBiasAdd&private__mlp_actor/pi/MatMul:product:04private__mlp_actor/pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pi/BiasAdd?
private__mlp_actor/pi/TanhTanh&private__mlp_actor/pi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pi/Tanhy
private__mlp_actor/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
private__mlp_actor/mul/y?
private__mlp_actor/mulMulprivate__mlp_actor/pi/Tanh:y:0!private__mlp_actor/mul/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mulu
IdentityIdentityprivate__mlp_actor/mul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp6^private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5^private__mlp_actor/actor/dense/MatMul/ReadVariableOp@^private__mlp_actor/actor/layer_normalization/add/ReadVariableOpB^private__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOpC^private__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOpB^private__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp-^private__mlp_actor/pi/BiasAdd/ReadVariableOp,^private__mlp_actor/pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2n
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp2l
4private__mlp_actor/actor/dense/MatMul/ReadVariableOp4private__mlp_actor/actor/dense/MatMul/ReadVariableOp2?
?private__mlp_actor/actor/layer_normalization/add/ReadVariableOp?private__mlp_actor/actor/layer_normalization/add/ReadVariableOp2?
Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOpAprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp2?
Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOpBprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp2?
Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOpAprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp2\
,private__mlp_actor/pi/BiasAdd/ReadVariableOp,private__mlp_actor/pi/BiasAdd/ReadVariableOp2Z
+private__mlp_actor/pi/MatMul/ReadVariableOp+private__mlp_actor/pi/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356042
x 
actor_46356009:o@
actor_46356011:@
actor_46356013:@
actor_46356015:@ 
actor_46356017:@@
actor_46356019:@
pi_46356034:@
pi_46356036:
identity??actor/StatefulPartitionedCall?pi/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_46356009actor_46356011actor_46356013actor_46356015actor_46356017actor_46356019*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463558502
actor/StatefulPartitionedCall?
pi/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0pi_46356034pi_46356036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_463560332
pi/StatefulPartitionedCallS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/yx
mulMul#pi/StatefulPartitionedCall:output:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^actor/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?&
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_46356828

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46356858

inputs8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Reluu
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
!__inference__traced_save_46356954
file_prefix(
$savev2_pi_kernel_read_readvariableop&
"savev2_pi_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_pi/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_pi_kernel_read_readvariableop"savev2_pi_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*S
_input_shapesB
@: :@::o@:@:@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:o@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:	

_output_shapes
: 
?
?
C__inference_actor_layer_call_and_return_conditional_losses_46355930

inputs 
dense_46355913:o@
dense_46355915:@*
layer_normalization_46355918:@*
layer_normalization_46355920:@%
sequential_46355924:@@!
sequential_46355926:@
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_46355913dense_46355915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_463557792
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_46355918layer_normalization_46355920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_463558312-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_463558422
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_46355924sequential_46355926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463557282$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_46355744
dense_1_input
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463557282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????@
'
_user_specified_namedense_1_input
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356436
input_1<
*actor_dense_matmul_readvariableop_resource:o@9
+actor_dense_biasadd_readvariableop_resource:@E
7actor_layer_normalization_mul_2_readvariableop_resource:@C
5actor_layer_normalization_add_readvariableop_resource:@I
7actor_sequential_dense_1_matmul_readvariableop_resource:@@F
8actor_sequential_dense_1_biasadd_readvariableop_resource:@3
!pi_matmul_readvariableop_resource:@0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulinput_1)actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/BiasAdd?
actor/layer_normalization/ShapeShapeactor/dense/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/layer_normalization/Shape?
-actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/layer_normalization/strided_slice/stack?
/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_1?
/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_2?
'actor/layer_normalization/strided_sliceStridedSlice(actor/layer_normalization/Shape:output:06actor/layer_normalization/strided_slice/stack:output:08actor/layer_normalization/strided_slice/stack_1:output:08actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'actor/layer_normalization/strided_slice?
actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
actor/layer_normalization/mul/x?
actor/layer_normalization/mulMul(actor/layer_normalization/mul/x:output:00actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
actor/layer_normalization/mul?
/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice_1/stack?
1actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_1?
1actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_2?
)actor/layer_normalization/strided_slice_1StridedSlice(actor/layer_normalization/Shape:output:08actor/layer_normalization/strided_slice_1/stack:output:0:actor/layer_normalization/strided_slice_1/stack_1:output:0:actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)actor/layer_normalization/strided_slice_1?
!actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!actor/layer_normalization/mul_1/x?
actor/layer_normalization/mul_1Mul*actor/layer_normalization/mul_1/x:output:02actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
actor/layer_normalization/mul_1?
)actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/0?
)actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/3?
'actor/layer_normalization/Reshape/shapePack2actor/layer_normalization/Reshape/shape/0:output:0!actor/layer_normalization/mul:z:0#actor/layer_normalization/mul_1:z:02actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'actor/layer_normalization/Reshape/shape?
!actor/layer_normalization/ReshapeReshapeactor/dense/BiasAdd:output:00actor/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!actor/layer_normalization/Reshape?
%actor/layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%actor/layer_normalization/ones/Less/y?
#actor/layer_normalization/ones/LessLess!actor/layer_normalization/mul:z:0.actor/layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2%
#actor/layer_normalization/ones/Less?
%actor/layer_normalization/ones/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%actor/layer_normalization/ones/packed?
$actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$actor/layer_normalization/ones/Const?
actor/layer_normalization/onesFill.actor/layer_normalization/ones/packed:output:0-actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2 
actor/layer_normalization/ones?
&actor/layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&actor/layer_normalization/zeros/Less/y?
$actor/layer_normalization/zeros/LessLess!actor/layer_normalization/mul:z:0/actor/layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$actor/layer_normalization/zeros/Less?
&actor/layer_normalization/zeros/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2(
&actor/layer_normalization/zeros/packed?
%actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%actor/layer_normalization/zeros/Const?
actor/layer_normalization/zerosFill/actor/layer_normalization/zeros/packed:output:0.actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2!
actor/layer_normalization/zeros?
actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
actor/layer_normalization/Const?
!actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!actor/layer_normalization/Const_1?
*actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3*actor/layer_normalization/Reshape:output:0'actor/layer_normalization/ones:output:0(actor/layer_normalization/zeros:output:0(actor/layer_normalization/Const:output:0*actor/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2,
*actor/layer_normalization/FusedBatchNormV3?
#actor/layer_normalization/Reshape_1Reshape.actor/layer_normalization/FusedBatchNormV3:y:0(actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
pi/TanhS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/y`
mulMulpi/Tanh:y:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2\
,actor/layer_normalization/add/ReadVariableOp,actor/layer_normalization/add/ReadVariableOp2`
.actor/layer_normalization/mul_2/ReadVariableOp.actor/layer_normalization/mul_2/ReadVariableOp2b
/actor/sequential/dense_1/BiasAdd/ReadVariableOp/actor/sequential/dense_1/BiasAdd/ReadVariableOp2`
.actor/sequential/dense_1/MatMul/ReadVariableOp.actor/sequential/dense_1/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
*__inference_dense_1_layer_call_fn_46356907

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_463556842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_layer_call_and_return_conditional_losses_46356773

inputs0
matmul_readvariableop_resource:o@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:o@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????o: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_layer_call_fn_46356837

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_463558312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
5__inference_private__mlp_actor_layer_call_fn_46356526
input_1
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_463560422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46355728

inputs"
dense_1_46355722:@@
dense_1_46355724:@
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_46355722dense_1_46355724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_463556842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_46355698
dense_1_input
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463556912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????@
'
_user_specified_namedense_1_input
?	
?
(__inference_actor_layer_call_fn_46355865
dense_input
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463558502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?
d
H__inference_activation_layer_call_and_return_conditional_losses_46356842

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:?????????@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?N
?
C__inference_actor_layer_call_and_return_conditional_losses_46356649

inputs6
$dense_matmul_readvariableop_resource:o@3
%dense_biasadd_readvariableop_resource:@?
1layer_normalization_mul_2_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@@
2sequential_dense_1_biasadd_readvariableop_resource:@
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd|
layer_normalization/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshapedense/BiasAdd:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape?
layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
layer_normalization/ones/Less/y?
layer_normalization/ones/LessLesslayer_normalization/mul:z:0(layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2
layer_normalization/ones/Less?
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/ones/packed?
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
layer_normalization/ones/Const?
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/ones?
 layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 layer_normalization/zeros/Less/y?
layer_normalization/zeros/LessLesslayer_normalization/mul:z:0)layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
layer_normalization/zeros/Less?
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2"
 layer_normalization/zeros/packed?
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer_normalization/zeros/Const?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/zerosy
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const}
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_1?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/addy
activation/TanhTanhlayer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
activation/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulactivation/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/Relu?
IdentityIdentity%sequential/dense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?	
?
(__inference_actor_layer_call_fn_46356726

inputs
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463558502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?

?
C__inference_dense_layer_call_and_return_conditional_losses_46355779

inputs0
matmul_readvariableop_resource:o@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:o@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????o: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46355753
dense_1_input"
dense_1_46355747:@@
dense_1_46355749:@
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_46355747dense_1_46355749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_463556842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????@
'
_user_specified_namedense_1_input
?
?
-__inference_sequential_layer_call_fn_46356878

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463556912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?&
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_46355831

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_46355684

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_46356229
input_1
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_463556662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46355762
dense_1_input"
dense_1_46355756:@@
dense_1_46355758:@
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_46355756dense_1_46355758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_463556842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????@
'
_user_specified_namedense_1_input
?
?
C__inference_actor_layer_call_and_return_conditional_losses_46356002
dense_input 
dense_46355985:o@
dense_46355987:@*
layer_normalization_46355990:@*
layer_normalization_46355992:@%
sequential_46355996:@@!
sequential_46355998:@
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_46355985dense_46355987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_463557792
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_46355990layer_normalization_46355992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_463558312-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_463558422
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_46355996sequential_46355998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463557282$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?	
?
5__inference_private__mlp_actor_layer_call_fn_46356568
x
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_463561182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?
d
H__inference_activation_layer_call_and_return_conditional_losses_46355842

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:?????????@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356505
input_1<
*actor_dense_matmul_readvariableop_resource:o@9
+actor_dense_biasadd_readvariableop_resource:@E
7actor_layer_normalization_mul_2_readvariableop_resource:@C
5actor_layer_normalization_add_readvariableop_resource:@I
7actor_sequential_dense_1_matmul_readvariableop_resource:@@F
8actor_sequential_dense_1_biasadd_readvariableop_resource:@3
!pi_matmul_readvariableop_resource:@0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulinput_1)actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/BiasAdd?
actor/layer_normalization/ShapeShapeactor/dense/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/layer_normalization/Shape?
-actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/layer_normalization/strided_slice/stack?
/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_1?
/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_2?
'actor/layer_normalization/strided_sliceStridedSlice(actor/layer_normalization/Shape:output:06actor/layer_normalization/strided_slice/stack:output:08actor/layer_normalization/strided_slice/stack_1:output:08actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'actor/layer_normalization/strided_slice?
actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
actor/layer_normalization/mul/x?
actor/layer_normalization/mulMul(actor/layer_normalization/mul/x:output:00actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
actor/layer_normalization/mul?
/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice_1/stack?
1actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_1?
1actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_2?
)actor/layer_normalization/strided_slice_1StridedSlice(actor/layer_normalization/Shape:output:08actor/layer_normalization/strided_slice_1/stack:output:0:actor/layer_normalization/strided_slice_1/stack_1:output:0:actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)actor/layer_normalization/strided_slice_1?
!actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!actor/layer_normalization/mul_1/x?
actor/layer_normalization/mul_1Mul*actor/layer_normalization/mul_1/x:output:02actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
actor/layer_normalization/mul_1?
)actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/0?
)actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/3?
'actor/layer_normalization/Reshape/shapePack2actor/layer_normalization/Reshape/shape/0:output:0!actor/layer_normalization/mul:z:0#actor/layer_normalization/mul_1:z:02actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'actor/layer_normalization/Reshape/shape?
!actor/layer_normalization/ReshapeReshapeactor/dense/BiasAdd:output:00actor/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!actor/layer_normalization/Reshape?
%actor/layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%actor/layer_normalization/ones/Less/y?
#actor/layer_normalization/ones/LessLess!actor/layer_normalization/mul:z:0.actor/layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2%
#actor/layer_normalization/ones/Less?
%actor/layer_normalization/ones/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%actor/layer_normalization/ones/packed?
$actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$actor/layer_normalization/ones/Const?
actor/layer_normalization/onesFill.actor/layer_normalization/ones/packed:output:0-actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2 
actor/layer_normalization/ones?
&actor/layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&actor/layer_normalization/zeros/Less/y?
$actor/layer_normalization/zeros/LessLess!actor/layer_normalization/mul:z:0/actor/layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$actor/layer_normalization/zeros/Less?
&actor/layer_normalization/zeros/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2(
&actor/layer_normalization/zeros/packed?
%actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%actor/layer_normalization/zeros/Const?
actor/layer_normalization/zerosFill/actor/layer_normalization/zeros/packed:output:0.actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2!
actor/layer_normalization/zeros?
actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
actor/layer_normalization/Const?
!actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!actor/layer_normalization/Const_1?
*actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3*actor/layer_normalization/Reshape:output:0'actor/layer_normalization/ones:output:0(actor/layer_normalization/zeros:output:0(actor/layer_normalization/Const:output:0*actor/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2,
*actor/layer_normalization/FusedBatchNormV3?
#actor/layer_normalization/Reshape_1Reshape.actor/layer_normalization/FusedBatchNormV3:y:0(actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
pi/TanhS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/y`
mulMulpi/Tanh:y:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2\
,actor/layer_normalization/add/ReadVariableOp,actor/layer_normalization/add/ReadVariableOp2`
.actor/layer_normalization/mul_2/ReadVariableOp.actor/layer_normalization/mul_2/ReadVariableOp2b
/actor/sequential/dense_1/BiasAdd/ReadVariableOp/actor/sequential/dense_1/BiasAdd/ReadVariableOp2`
.actor/sequential/dense_1/MatMul/ReadVariableOp.actor/sequential/dense_1/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?

?
@__inference_pi_layer_call_and_return_conditional_losses_46356033

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46355691

inputs"
dense_1_46355685:@@
dense_1_46355687:@
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_46355685dense_1_46355687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_463556842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356298
x<
*actor_dense_matmul_readvariableop_resource:o@9
+actor_dense_biasadd_readvariableop_resource:@E
7actor_layer_normalization_mul_2_readvariableop_resource:@C
5actor_layer_normalization_add_readvariableop_resource:@I
7actor_sequential_dense_1_matmul_readvariableop_resource:@@F
8actor_sequential_dense_1_biasadd_readvariableop_resource:@3
!pi_matmul_readvariableop_resource:@0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulx)actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/BiasAdd?
actor/layer_normalization/ShapeShapeactor/dense/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/layer_normalization/Shape?
-actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/layer_normalization/strided_slice/stack?
/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_1?
/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_2?
'actor/layer_normalization/strided_sliceStridedSlice(actor/layer_normalization/Shape:output:06actor/layer_normalization/strided_slice/stack:output:08actor/layer_normalization/strided_slice/stack_1:output:08actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'actor/layer_normalization/strided_slice?
actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
actor/layer_normalization/mul/x?
actor/layer_normalization/mulMul(actor/layer_normalization/mul/x:output:00actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
actor/layer_normalization/mul?
/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice_1/stack?
1actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_1?
1actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_2?
)actor/layer_normalization/strided_slice_1StridedSlice(actor/layer_normalization/Shape:output:08actor/layer_normalization/strided_slice_1/stack:output:0:actor/layer_normalization/strided_slice_1/stack_1:output:0:actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)actor/layer_normalization/strided_slice_1?
!actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!actor/layer_normalization/mul_1/x?
actor/layer_normalization/mul_1Mul*actor/layer_normalization/mul_1/x:output:02actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
actor/layer_normalization/mul_1?
)actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/0?
)actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/3?
'actor/layer_normalization/Reshape/shapePack2actor/layer_normalization/Reshape/shape/0:output:0!actor/layer_normalization/mul:z:0#actor/layer_normalization/mul_1:z:02actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'actor/layer_normalization/Reshape/shape?
!actor/layer_normalization/ReshapeReshapeactor/dense/BiasAdd:output:00actor/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!actor/layer_normalization/Reshape?
%actor/layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%actor/layer_normalization/ones/Less/y?
#actor/layer_normalization/ones/LessLess!actor/layer_normalization/mul:z:0.actor/layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2%
#actor/layer_normalization/ones/Less?
%actor/layer_normalization/ones/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%actor/layer_normalization/ones/packed?
$actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$actor/layer_normalization/ones/Const?
actor/layer_normalization/onesFill.actor/layer_normalization/ones/packed:output:0-actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2 
actor/layer_normalization/ones?
&actor/layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&actor/layer_normalization/zeros/Less/y?
$actor/layer_normalization/zeros/LessLess!actor/layer_normalization/mul:z:0/actor/layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$actor/layer_normalization/zeros/Less?
&actor/layer_normalization/zeros/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2(
&actor/layer_normalization/zeros/packed?
%actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%actor/layer_normalization/zeros/Const?
actor/layer_normalization/zerosFill/actor/layer_normalization/zeros/packed:output:0.actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2!
actor/layer_normalization/zeros?
actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
actor/layer_normalization/Const?
!actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!actor/layer_normalization/Const_1?
*actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3*actor/layer_normalization/Reshape:output:0'actor/layer_normalization/ones:output:0(actor/layer_normalization/zeros:output:0(actor/layer_normalization/Const:output:0*actor/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2,
*actor/layer_normalization/FusedBatchNormV3?
#actor/layer_normalization/Reshape_1Reshape.actor/layer_normalization/FusedBatchNormV3:y:0(actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
pi/TanhS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/y`
mulMulpi/Tanh:y:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2\
,actor/layer_normalization/add/ReadVariableOp,actor/layer_normalization/add/ReadVariableOp2`
.actor/layer_normalization/mul_2/ReadVariableOp.actor/layer_normalization/mul_2/ReadVariableOp2b
/actor/sequential/dense_1/BiasAdd/ReadVariableOp/actor/sequential/dense_1/BiasAdd/ReadVariableOp2`
.actor/sequential/dense_1/MatMul/ReadVariableOp.actor/sequential/dense_1/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????o

_user_specified_namex
?

?
5__inference_private__mlp_actor_layer_call_fn_46356589
input_1
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_463561182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?	
?
(__inference_actor_layer_call_fn_46355962
dense_input
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463559302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?
?
%__inference_pi_layer_call_fn_46356763

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_463560332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356118
x 
actor_46356097:o@
actor_46356099:@
actor_46356101:@
actor_46356103:@ 
actor_46356105:@@
actor_46356107:@
pi_46356110:@
pi_46356112:
identity??actor/StatefulPartitionedCall?pi/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_46356097actor_46356099actor_46356101actor_46356103actor_46356105actor_46356107*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463559302
actor/StatefulPartitionedCall?
pi/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0pi_46356110pi_46356112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_463560332
pi/StatefulPartitionedCallS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/yx
mulMul#pi/StatefulPartitionedCall:output:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^actor/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?	
?
5__inference_private__mlp_actor_layer_call_fn_46356547
x
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_463560422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?

?
@__inference_pi_layer_call_and_return_conditional_losses_46356754

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_46356782

inputs
unknown:o@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_463557792
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????o: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?&
?
$__inference__traced_restore_46356988
file_prefix,
assignvariableop_pi_kernel:@(
assignvariableop_1_pi_bias:1
assignvariableop_2_dense_kernel:o@+
assignvariableop_3_dense_bias:@:
,assignvariableop_4_layer_normalization_gamma:@9
+assignvariableop_5_layer_normalization_beta:@3
!assignvariableop_6_dense_1_kernel:@@-
assignvariableop_7_dense_1_bias:@

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_pi/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_pi_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_pi_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_layer_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_layer_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8c

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_9?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_actor_layer_call_and_return_conditional_losses_46355850

inputs 
dense_46355780:o@
dense_46355782:@*
layer_normalization_46355832:@*
layer_normalization_46355834:@%
sequential_46355844:@@!
sequential_46355846:@
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_46355780dense_46355782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_463557792
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_46355832layer_normalization_46355834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_463558312-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_463558422
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_46355844sequential_46355846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463556912$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?	
?
(__inference_actor_layer_call_fn_46356743

inputs
unknown:o@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_463559302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_46356887

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463557282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356367
x<
*actor_dense_matmul_readvariableop_resource:o@9
+actor_dense_biasadd_readvariableop_resource:@E
7actor_layer_normalization_mul_2_readvariableop_resource:@C
5actor_layer_normalization_add_readvariableop_resource:@I
7actor_sequential_dense_1_matmul_readvariableop_resource:@@F
8actor_sequential_dense_1_biasadd_readvariableop_resource:@3
!pi_matmul_readvariableop_resource:@0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes

:o@*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulx)actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/dense/BiasAdd?
actor/layer_normalization/ShapeShapeactor/dense/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/layer_normalization/Shape?
-actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/layer_normalization/strided_slice/stack?
/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_1?
/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice/stack_2?
'actor/layer_normalization/strided_sliceStridedSlice(actor/layer_normalization/Shape:output:06actor/layer_normalization/strided_slice/stack:output:08actor/layer_normalization/strided_slice/stack_1:output:08actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'actor/layer_normalization/strided_slice?
actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
actor/layer_normalization/mul/x?
actor/layer_normalization/mulMul(actor/layer_normalization/mul/x:output:00actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
actor/layer_normalization/mul?
/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/layer_normalization/strided_slice_1/stack?
1actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_1?
1actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/layer_normalization/strided_slice_1/stack_2?
)actor/layer_normalization/strided_slice_1StridedSlice(actor/layer_normalization/Shape:output:08actor/layer_normalization/strided_slice_1/stack:output:0:actor/layer_normalization/strided_slice_1/stack_1:output:0:actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)actor/layer_normalization/strided_slice_1?
!actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!actor/layer_normalization/mul_1/x?
actor/layer_normalization/mul_1Mul*actor/layer_normalization/mul_1/x:output:02actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
actor/layer_normalization/mul_1?
)actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/0?
)actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)actor/layer_normalization/Reshape/shape/3?
'actor/layer_normalization/Reshape/shapePack2actor/layer_normalization/Reshape/shape/0:output:0!actor/layer_normalization/mul:z:0#actor/layer_normalization/mul_1:z:02actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'actor/layer_normalization/Reshape/shape?
!actor/layer_normalization/ReshapeReshapeactor/dense/BiasAdd:output:00actor/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!actor/layer_normalization/Reshape?
%actor/layer_normalization/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%actor/layer_normalization/ones/Less/y?
#actor/layer_normalization/ones/LessLess!actor/layer_normalization/mul:z:0.actor/layer_normalization/ones/Less/y:output:0*
T0*
_output_shapes
: 2%
#actor/layer_normalization/ones/Less?
%actor/layer_normalization/ones/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%actor/layer_normalization/ones/packed?
$actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$actor/layer_normalization/ones/Const?
actor/layer_normalization/onesFill.actor/layer_normalization/ones/packed:output:0-actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????2 
actor/layer_normalization/ones?
&actor/layer_normalization/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&actor/layer_normalization/zeros/Less/y?
$actor/layer_normalization/zeros/LessLess!actor/layer_normalization/mul:z:0/actor/layer_normalization/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$actor/layer_normalization/zeros/Less?
&actor/layer_normalization/zeros/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2(
&actor/layer_normalization/zeros/packed?
%actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%actor/layer_normalization/zeros/Const?
actor/layer_normalization/zerosFill/actor/layer_normalization/zeros/packed:output:0.actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2!
actor/layer_normalization/zeros?
actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
actor/layer_normalization/Const?
!actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!actor/layer_normalization/Const_1?
*actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3*actor/layer_normalization/Reshape:output:0'actor/layer_normalization/ones:output:0(actor/layer_normalization/zeros:output:0(actor/layer_normalization/Const:output:0*actor/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2,
*actor/layer_normalization/FusedBatchNormV3?
#actor/layer_normalization/Reshape_1Reshape.actor/layer_normalization/FusedBatchNormV3:y:0(actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*'
_output_shapes
:?????????@2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
pi/TanhS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/y`
mulMulpi/Tanh:y:0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2\
,actor/layer_normalization/add/ReadVariableOp,actor/layer_normalization/add/ReadVariableOp2`
.actor/layer_normalization/mul_2/ReadVariableOp.actor/layer_normalization/mul_2/ReadVariableOp2b
/actor/sequential/dense_1/BiasAdd/ReadVariableOp/actor/sequential/dense_1/BiasAdd/ReadVariableOp2`
.actor/sequential/dense_1/MatMul/ReadVariableOp.actor/sequential/dense_1/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????o

_user_specified_namex
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_46356898

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_46356869

inputs8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Reluu
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_46355982
dense_input 
dense_46355965:o@
dense_46355967:@*
layer_normalization_46355970:@*
layer_normalization_46355972:@%
sequential_46355976:@@!
sequential_46355978:@
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_46355965dense_46355967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_463557792
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_46355970layer_normalization_46355972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_463558312-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_463558422
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_46355976sequential_46355978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_463556912$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????o: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?
I
-__inference_activation_layer_call_fn_46356847

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_463558422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????o<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ԍ
?
	_body
_pi
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*Z&call_and_return_all_conditional_losses
[_default_save_signature
\__call__"
_tf_keras_model
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer_with_weights-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"
_tf_keras_sequential
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?
layer_regularization_losses
layer_metrics
	variables
regularization_losses

layers
metrics
trainable_variables
 non_trainable_variables
\__call__
[_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
?

kernel
bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
?
%axis
	gamma
beta
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
?
*regularization_losses
+	variables
,trainable_variables
-	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
?
.layer_with_weights-0
.layer-0
/	variables
0regularization_losses
1trainable_variables
2	keras_api
*h&call_and_return_all_conditional_losses
i__call__"
_tf_keras_sequential
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
3layer_regularization_losses
4layer_metrics
	variables
regularization_losses

5layers
6metrics
trainable_variables
7non_trainable_variables
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:@2	pi/kernel
:2pi/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
8layer_regularization_losses
9layer_metrics
regularization_losses
	variables

:layers
;metrics
trainable_variables
<non_trainable_variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:o@2dense/kernel
:@2
dense/bias
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
 :@@2dense_1/kernel
:@2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
=layer_regularization_losses
>layer_metrics
!regularization_losses
"	variables

?layers
@metrics
#trainable_variables
Anon_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Blayer_regularization_losses
Clayer_metrics
&regularization_losses
'	variables

Dlayers
Emetrics
(trainable_variables
Fnon_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Glayer_regularization_losses
Hlayer_metrics
*regularization_losses
+	variables

Ilayers
Jmetrics
,trainable_variables
Knon_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?

kernel
bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
*j&call_and_return_all_conditional_losses
k__call__"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Player_regularization_losses
Qlayer_metrics
/	variables
0regularization_losses

Rlayers
Smetrics
1trainable_variables
Tnon_trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Ulayer_regularization_losses
Vlayer_metrics
Lregularization_losses
M	variables

Wlayers
Xmetrics
Ntrainable_variables
Ynon_trainable_variables
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356298
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356367
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356436
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356505?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__wrapped_model_46355666input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_private__mlp_actor_layer_call_fn_46356526
5__inference_private__mlp_actor_layer_call_fn_46356547
5__inference_private__mlp_actor_layer_call_fn_46356568
5__inference_private__mlp_actor_layer_call_fn_46356589?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_actor_layer_call_and_return_conditional_losses_46356649
C__inference_actor_layer_call_and_return_conditional_losses_46356709
C__inference_actor_layer_call_and_return_conditional_losses_46355982
C__inference_actor_layer_call_and_return_conditional_losses_46356002?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_actor_layer_call_fn_46355865
(__inference_actor_layer_call_fn_46356726
(__inference_actor_layer_call_fn_46356743
(__inference_actor_layer_call_fn_46355962?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_pi_layer_call_and_return_conditional_losses_46356754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_pi_layer_call_fn_46356763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_46356229input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_46356773?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_46356782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_46356828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_layer_normalization_layer_call_fn_46356837?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_layer_call_and_return_conditional_losses_46356842?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_layer_call_fn_46356847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_46356858
H__inference_sequential_layer_call_and_return_conditional_losses_46356869
H__inference_sequential_layer_call_and_return_conditional_losses_46355753
H__inference_sequential_layer_call_and_return_conditional_losses_46355762?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_layer_call_fn_46355698
-__inference_sequential_layer_call_fn_46356878
-__inference_sequential_layer_call_fn_46356887
-__inference_sequential_layer_call_fn_46355744?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_46356898?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_46356907?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_46355666q0?-
&?#
!?
input_1?????????o
? "3?0
.
output_1"?
output_1??????????
H__inference_activation_layer_call_and_return_conditional_losses_46356842X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? |
-__inference_activation_layer_call_fn_46356847K/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_actor_layer_call_and_return_conditional_losses_46355982m<?9
2?/
%?"
dense_input?????????o
p 

 
? "%?"
?
0?????????@
? ?
C__inference_actor_layer_call_and_return_conditional_losses_46356002m<?9
2?/
%?"
dense_input?????????o
p

 
? "%?"
?
0?????????@
? ?
C__inference_actor_layer_call_and_return_conditional_losses_46356649h7?4
-?*
 ?
inputs?????????o
p 

 
? "%?"
?
0?????????@
? ?
C__inference_actor_layer_call_and_return_conditional_losses_46356709h7?4
-?*
 ?
inputs?????????o
p

 
? "%?"
?
0?????????@
? ?
(__inference_actor_layer_call_fn_46355865`<?9
2?/
%?"
dense_input?????????o
p 

 
? "??????????@?
(__inference_actor_layer_call_fn_46355962`<?9
2?/
%?"
dense_input?????????o
p

 
? "??????????@?
(__inference_actor_layer_call_fn_46356726[7?4
-?*
 ?
inputs?????????o
p 

 
? "??????????@?
(__inference_actor_layer_call_fn_46356743[7?4
-?*
 ?
inputs?????????o
p

 
? "??????????@?
E__inference_dense_1_layer_call_and_return_conditional_losses_46356898\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_1_layer_call_fn_46356907O/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_dense_layer_call_and_return_conditional_losses_46356773\/?,
%?"
 ?
inputs?????????o
? "%?"
?
0?????????@
? {
(__inference_dense_layer_call_fn_46356782O/?,
%?"
 ?
inputs?????????o
? "??????????@?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_46356828\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
6__inference_layer_normalization_layer_call_fn_46356837O/?,
%?"
 ?
inputs?????????@
? "??????????@?
@__inference_pi_layer_call_and_return_conditional_losses_46356754\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? x
%__inference_pi_layer_call_fn_46356763O/?,
%?"
 ?
inputs?????????@
? "???????????
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356298a.?+
$?!
?
x?????????o
p 
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356367a.?+
$?!
?
x?????????o
p
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356436g4?1
*?'
!?
input_1?????????o
p 
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_46356505g4?1
*?'
!?
input_1?????????o
p
? "%?"
?
0?????????
? ?
5__inference_private__mlp_actor_layer_call_fn_46356526Z4?1
*?'
!?
input_1?????????o
p 
? "???????????
5__inference_private__mlp_actor_layer_call_fn_46356547T.?+
$?!
?
x?????????o
p 
? "???????????
5__inference_private__mlp_actor_layer_call_fn_46356568T.?+
$?!
?
x?????????o
p
? "???????????
5__inference_private__mlp_actor_layer_call_fn_46356589Z4?1
*?'
!?
input_1?????????o
p
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_46355753k>?;
4?1
'?$
dense_1_input?????????@
p 

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_46355762k>?;
4?1
'?$
dense_1_input?????????@
p

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_46356858d7?4
-?*
 ?
inputs?????????@
p 

 
? "%?"
?
0?????????@
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_46356869d7?4
-?*
 ?
inputs?????????@
p

 
? "%?"
?
0?????????@
? ?
-__inference_sequential_layer_call_fn_46355698^>?;
4?1
'?$
dense_1_input?????????@
p 

 
? "??????????@?
-__inference_sequential_layer_call_fn_46355744^>?;
4?1
'?$
dense_1_input?????????@
p

 
? "??????????@?
-__inference_sequential_layer_call_fn_46356878W7?4
-?*
 ?
inputs?????????@
p 

 
? "??????????@?
-__inference_sequential_layer_call_fn_46356887W7?4
-?*
 ?
inputs?????????@
p

 
? "??????????@?
&__inference_signature_wrapper_46356229|;?8
? 
1?.
,
input_1!?
input_1?????????o"3?0
.
output_1"?
output_1?????????