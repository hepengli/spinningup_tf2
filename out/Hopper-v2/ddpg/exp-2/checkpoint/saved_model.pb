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

o
	pi/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name	pi/kernel
h
pi/kernel/Read/ReadVariableOpReadVariableOp	pi/kernel*
_output_shapes
:	?*
dtype0
f
pi/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pi/bias
_
pi/bias/Read/ReadVariableOpReadVariableOppi/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
v
	_body
_pi
trainable_variables
regularization_losses
	variables
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
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
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
trainable_variables
regularization_losses
	variables
non_trainable_variables
metrics

layers
 layer_metrics
 
h

kernel
bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
q
%axis
	gamma
beta
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api
y
.layer_with_weights-0
.layer-0
/trainable_variables
0regularization_losses
1	variables
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
trainable_variables
regularization_losses
	variables
4non_trainable_variables
5metrics

6layers
7layer_metrics
DB
VARIABLE_VALUE	pi/kernel%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUE
@>
VARIABLE_VALUEpi/bias#_pi/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
8layer_regularization_losses
trainable_variables
regularization_losses
	variables
9non_trainable_variables
:metrics

;layers
<layer_metrics
RP
VARIABLE_VALUEdense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElayer_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElayer_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1
 

0
1
?
=layer_regularization_losses
!trainable_variables
"regularization_losses
#	variables
>non_trainable_variables
?metrics

@layers
Alayer_metrics
 

0
1
 

0
1
?
Blayer_regularization_losses
&trainable_variables
'regularization_losses
(	variables
Cnon_trainable_variables
Dmetrics

Elayers
Flayer_metrics
 
 
 
?
Glayer_regularization_losses
*trainable_variables
+regularization_losses
,	variables
Hnon_trainable_variables
Imetrics

Jlayers
Klayer_metrics
h

kernel
bias
Ltrainable_variables
Mregularization_losses
N	variables
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
/trainable_variables
0regularization_losses
1	variables
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
 
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

0
1
 

0
1
?
Ulayer_regularization_losses
Ltrainable_variables
Mregularization_losses
N	variables
Vnon_trainable_variables
Wmetrics

Xlayers
Ylayer_metrics
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
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
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_13432133
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
!__inference__traced_save_13432858
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
$__inference__traced_restore_13432892??	
?
?
-__inference_sequential_layer_call_fn_13431648
dense_1_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134316322
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_1_input
?N
?
C__inference_actor_layer_call_and_return_conditional_losses_13432613

inputs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?@
1layer_normalization_mul_2_readvariableop_resource:	?>
/layer_normalization_add_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/addz
activation/TanhTanhlayer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
activation/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulactivation/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
IdentityIdentity%sequential/dense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
!__inference__traced_save_13432858
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_pi/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

identity_1Identity_1:output:0*[
_input_shapesJ
H: :	?::	?:?:?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:	

_output_shapes
: 
?	
?
5__inference_private__mlp_actor_layer_call_fn_13432472
x
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_134320222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?
?
6__inference_layer_normalization_layer_call_fn_13432741

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_134317352
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_13432773

inputs:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Reluv
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_13431754

inputs!
dense_13431684:	?
dense_13431686:	?+
layer_normalization_13431736:	?+
layer_normalization_13431738:	?'
sequential_13431748:
??"
sequential_13431750:	?
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13431684dense_13431686*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_134316832
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_13431736layer_normalization_13431738*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_134317352-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_134317462
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_13431748sequential_13431750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134315952$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
5__inference_private__mlp_actor_layer_call_fn_13432430
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_134319462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_dense_1_layer_call_fn_13432811

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_134315882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_13431906
dense_input!
dense_13431889:	?
dense_13431891:	?+
layer_normalization_13431894:	?+
layer_normalization_13431896:	?'
sequential_13431900:
??"
sequential_13431902:	?
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_13431889dense_13431891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_134316832
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_13431894layer_normalization_13431896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_134317352-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_134317462
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_13431900sequential_13431902*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134316322$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?	
?
(__inference_actor_layer_call_fn_13431769
dense_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134317542
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432022
x!
actor_13432001:	?
actor_13432003:	?
actor_13432005:	?
actor_13432007:	?"
actor_13432009:
??
actor_13432011:	?
pi_13432014:	?
pi_13432016:
identity??actor/StatefulPartitionedCall?pi/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_13432001actor_13432003actor_13432005actor_13432007actor_13432009actor_13432011*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134318342
actor/StatefulPartitionedCall?
pi/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0pi_13432014pi_13432016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_134319372
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^actor/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?

?
C__inference_dense_layer_call_and_return_conditional_losses_13432677

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_13431735

inputs,
mul_2_readvariableop_resource:	?*
add_readvariableop_resource:	?
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
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
addc
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
(__inference_actor_layer_call_fn_13432647

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134318342
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432409
input_1=
*actor_dense_matmul_readvariableop_resource:	?:
+actor_dense_biasadd_readvariableop_resource:	?F
7actor_layer_normalization_mul_2_readvariableop_resource:	?D
5actor_layer_normalization_add_readvariableop_resource:	?K
7actor_sequential_dense_1_matmul_readvariableop_resource:
??G
8actor_sequential_dense_1_biasadd_readvariableop_resource:	?4
!pi_matmul_readvariableop_resource:	?0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulinput_1)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
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
:?????????
!
_user_specified_name	input_1
?

?
C__inference_dense_layer_call_and_return_conditional_losses_13431683

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_pi_layer_call_and_return_conditional_losses_13431937

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_layer_call_fn_13432751

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_134317462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_13432133
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_134315702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
5__inference_private__mlp_actor_layer_call_fn_13432451
x
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_134319462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_13432762

inputs:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Reluv
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_13432791

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134316322
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
(__inference_actor_layer_call_fn_13431866
dense_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134318342
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_13431632

inputs$
dense_1_13431626:
??
dense_1_13431628:	?
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_13431626dense_1_13431628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_134315882!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_pi_layer_call_and_return_conditional_losses_13432658

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_layer_call_and_return_conditional_losses_13431746

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:??????????2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432202
x=
*actor_dense_matmul_readvariableop_resource:	?:
+actor_dense_biasadd_readvariableop_resource:	?F
7actor_layer_normalization_mul_2_readvariableop_resource:	?D
5actor_layer_normalization_add_readvariableop_resource:	?K
7actor_sequential_dense_1_matmul_readvariableop_resource:
??G
8actor_sequential_dense_1_biasadd_readvariableop_resource:	?4
!pi_matmul_readvariableop_resource:	?0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulx)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
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
:?????????

_user_specified_namex
?
?
C__inference_actor_layer_call_and_return_conditional_losses_13431834

inputs!
dense_13431817:	?
dense_13431819:	?+
layer_normalization_13431822:	?+
layer_normalization_13431824:	?'
sequential_13431828:
??"
sequential_13431830:	?
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13431817dense_13431819*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_134316832
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_13431822layer_normalization_13431824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_134317352-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_134317462
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_13431828sequential_13431830*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134316322$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_activation_layer_call_and_return_conditional_losses_13432746

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:??????????2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_13432686

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_134316832
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_sequential_layer_call_and_return_conditional_losses_13431666
dense_1_input$
dense_1_13431660:
??
dense_1_13431662:	?
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_13431660dense_1_13431662*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_134315882!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_1_input
?
?
C__inference_actor_layer_call_and_return_conditional_losses_13431886
dense_input!
dense_13431869:	?
dense_13431871:	?+
layer_normalization_13431874:	?+
layer_normalization_13431876:	?'
sequential_13431880:
??"
sequential_13431882:	?
identity??dense/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_13431869dense_13431871*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_134316832
dense/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_13431874layer_normalization_13431876*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_134317352-
+layer_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_134317462
activation/PartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0sequential_13431880sequential_13431882*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134315952$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?
?
-__inference_sequential_layer_call_fn_13432782

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134315952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13431946
x!
actor_13431913:	?
actor_13431915:	?
actor_13431917:	?
actor_13431919:	?"
actor_13431921:
??
actor_13431923:	?
pi_13431938:	?
pi_13431940:
identity??actor/StatefulPartitionedCall?pi/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_13431913actor_13431915actor_13431917actor_13431919actor_13431921actor_13431923*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134317542
actor/StatefulPartitionedCall?
pi/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0pi_13431938pi_13431940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_134319372
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^actor/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_namex
?'
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_13432732

inputs,
mul_2_readvariableop_resource:	?*
add_readvariableop_resource:	?
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
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
addc
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_sequential_layer_call_and_return_conditional_losses_13431657
dense_1_input$
dense_1_13431651:
??
dense_1_13431653:	?
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_13431651dense_1_13431653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_134315882!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_1_input
?
?
-__inference_sequential_layer_call_fn_13431602
dense_1_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_134315952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_1_input
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_13431595

inputs$
dense_1_13431589:
??
dense_1_13431591:	?
identity??dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_13431589dense_1_13431591*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_134315882!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_13432802

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_13431588

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
C__inference_actor_layer_call_and_return_conditional_losses_13432553

inputs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?@
1layer_normalization_mul_2_readvariableop_resource:	?>
/layer_normalization_add_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/addz
activation/TanhTanhlayer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
activation/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulactivation/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
IdentityIdentity%sequential/dense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?~
?	
#__inference__wrapped_model_13431570
input_1P
=private__mlp_actor_actor_dense_matmul_readvariableop_resource:	?M
>private__mlp_actor_actor_dense_biasadd_readvariableop_resource:	?Y
Jprivate__mlp_actor_actor_layer_normalization_mul_2_readvariableop_resource:	?W
Hprivate__mlp_actor_actor_layer_normalization_add_readvariableop_resource:	?^
Jprivate__mlp_actor_actor_sequential_dense_1_matmul_readvariableop_resource:
??Z
Kprivate__mlp_actor_actor_sequential_dense_1_biasadd_readvariableop_resource:	?G
4private__mlp_actor_pi_matmul_readvariableop_resource:	?C
5private__mlp_actor_pi_biasadd_readvariableop_resource:
identity??5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp?4private__mlp_actor/actor/dense/MatMul/ReadVariableOp??private__mlp_actor/actor/layer_normalization/add/ReadVariableOp?Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp?Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp?Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp?,private__mlp_actor/pi/BiasAdd/ReadVariableOp?+private__mlp_actor/pi/MatMul/ReadVariableOp?
4private__mlp_actor/actor/dense/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_actor_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4private__mlp_actor/actor/dense/MatMul/ReadVariableOp?
%private__mlp_actor/actor/dense/MatMulMatMulinput_1<private__mlp_actor/actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%private__mlp_actor/actor/dense/MatMul?
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOpReadVariableOp>private__mlp_actor_actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp?
&private__mlp_actor/actor/dense/BiasAddBiasAdd/private__mlp_actor/actor/dense/MatMul:product:0=private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
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
T0*(
_output_shapes
:??????????28
6private__mlp_actor/actor/layer_normalization/Reshape_1?
Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOpReadVariableOpJprivate__mlp_actor_actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Aprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp?
2private__mlp_actor/actor/layer_normalization/mul_2Mul?private__mlp_actor/actor/layer_normalization/Reshape_1:output:0Iprivate__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2private__mlp_actor/actor/layer_normalization/mul_2?
?private__mlp_actor/actor/layer_normalization/add/ReadVariableOpReadVariableOpHprivate__mlp_actor_actor_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?private__mlp_actor/actor/layer_normalization/add/ReadVariableOp?
0private__mlp_actor/actor/layer_normalization/addAddV26private__mlp_actor/actor/layer_normalization/mul_2:z:0Gprivate__mlp_actor/actor/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0private__mlp_actor/actor/layer_normalization/add?
(private__mlp_actor/actor/activation/TanhTanh4private__mlp_actor/actor/layer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2*
(private__mlp_actor/actor/activation/Tanh?
Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOpJprivate__mlp_actor_actor_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02C
Aprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp?
2private__mlp_actor/actor/sequential/dense_1/MatMulMatMul,private__mlp_actor/actor/activation/Tanh:y:0Iprivate__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2private__mlp_actor/actor/sequential/dense_1/MatMul?
Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpKprivate__mlp_actor_actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
3private__mlp_actor/actor/sequential/dense_1/BiasAddBiasAdd<private__mlp_actor/actor/sequential/dense_1/MatMul:product:0Jprivate__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3private__mlp_actor/actor/sequential/dense_1/BiasAdd?
0private__mlp_actor/actor/sequential/dense_1/ReluRelu<private__mlp_actor/actor/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????22
0private__mlp_actor/actor/sequential/dense_1/Relu?
+private__mlp_actor/pi/MatMul/ReadVariableOpReadVariableOp4private__mlp_actor_pi_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+private__mlp_actor/pi/MatMul/ReadVariableOp?
private__mlp_actor/pi/MatMulMatMul>private__mlp_actor/actor/sequential/dense_1/Relu:activations:03private__mlp_actor/pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pi/MatMul?
,private__mlp_actor/pi/BiasAdd/ReadVariableOpReadVariableOp5private__mlp_actor_pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,private__mlp_actor/pi/BiasAdd/ReadVariableOp?
private__mlp_actor/pi/BiasAddBiasAdd&private__mlp_actor/pi/MatMul:product:04private__mlp_actor/pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pi/BiasAdd?
private__mlp_actor/pi/TanhTanh&private__mlp_actor/pi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
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
:?????????2
private__mlp_actor/mulu
IdentityIdentityprivate__mlp_actor/mul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp6^private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5^private__mlp_actor/actor/dense/MatMul/ReadVariableOp@^private__mlp_actor/actor/layer_normalization/add/ReadVariableOpB^private__mlp_actor/actor/layer_normalization/mul_2/ReadVariableOpC^private__mlp_actor/actor/sequential/dense_1/BiasAdd/ReadVariableOpB^private__mlp_actor/actor/sequential/dense_1/MatMul/ReadVariableOp-^private__mlp_actor/pi/BiasAdd/ReadVariableOp,^private__mlp_actor/pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2n
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
:?????????
!
_user_specified_name	input_1
?
?
%__inference_pi_layer_call_fn_13432667

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_pi_layer_call_and_return_conditional_losses_134319372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432271
x=
*actor_dense_matmul_readvariableop_resource:	?:
+actor_dense_biasadd_readvariableop_resource:	?F
7actor_layer_normalization_mul_2_readvariableop_resource:	?D
5actor_layer_normalization_add_readvariableop_resource:	?K
7actor_sequential_dense_1_matmul_readvariableop_resource:
??G
8actor_sequential_dense_1_biasadd_readvariableop_resource:	?4
!pi_matmul_readvariableop_resource:	?0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulx)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
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
:?????????

_user_specified_namex
?

?
5__inference_private__mlp_actor_layer_call_fn_13432493
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_134320222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
(__inference_actor_layer_call_fn_13432630

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_134317542
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
$__inference__traced_restore_13432892
file_prefix-
assignvariableop_pi_kernel:	?(
assignvariableop_1_pi_bias:2
assignvariableop_2_dense_kernel:	?,
assignvariableop_3_dense_bias:	?;
,assignvariableop_4_layer_normalization_gamma:	?:
+assignvariableop_5_layer_normalization_beta:	?5
!assignvariableop_6_dense_1_kernel:
??.
assignvariableop_7_dense_1_bias:	?

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%_pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_pi/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
?_
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432340
input_1=
*actor_dense_matmul_readvariableop_resource:	?:
+actor_dense_biasadd_readvariableop_resource:	?F
7actor_layer_normalization_mul_2_readvariableop_resource:	?D
5actor_layer_normalization_add_readvariableop_resource:	?K
7actor_sequential_dense_1_matmul_readvariableop_resource:
??G
8actor_sequential_dense_1_biasadd_readvariableop_resource:	?4
!pi_matmul_readvariableop_resource:	?0
"pi_biasadd_readvariableop_resource:
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?/actor/sequential/dense_1/BiasAdd/ReadVariableOp?.actor/sequential/dense_1/MatMul/ReadVariableOp?pi/BiasAdd/ReadVariableOp?pi/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulinput_1)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
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
T0*(
_output_shapes
:??????????2%
#actor/layer_normalization/Reshape_1?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype020
.actor/layer_normalization/mul_2/ReadVariableOp?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/layer_normalization/mul_2?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,actor/layer_normalization/add/ReadVariableOp?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/layer_normalization/add?
actor/activation/TanhTanh!actor/layer_normalization/add:z:0*
T0*(
_output_shapes
:??????????2
actor/activation/Tanh?
.actor/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7actor_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.actor/sequential/dense_1/MatMul/ReadVariableOp?
actor/sequential/dense_1/MatMulMatMulactor/activation/Tanh:y:06actor/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
actor/sequential/dense_1/MatMul?
/actor/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8actor_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/actor/sequential/dense_1/BiasAdd/ReadVariableOp?
 actor/sequential/dense_1/BiasAddBiasAdd)actor/sequential/dense_1/MatMul:product:07actor/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 actor/sequential/dense_1/BiasAdd?
actor/sequential/dense_1/ReluRelu)actor/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/sequential/dense_1/Relu?
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
pi/MatMul/ReadVariableOp?
	pi/MatMulMatMul+actor/sequential/dense_1/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pi/MatMul?
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOp?

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

pi/BiasAdda
pi/TanhTanhpi/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
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
:?????????2
mulb
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp0^actor/sequential/dense_1/BiasAdd/ReadVariableOp/^actor/sequential/dense_1/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
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
:?????????
!
_user_specified_name	input_1"?L
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
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	_body
_pi
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*Z&call_and_return_all_conditional_losses
[__call__
\_default_save_signature"
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
trainable_variables
regularization_losses
	variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"
_tf_keras_sequential
?

kernel
bias
trainable_variables
regularization_losses
	variables
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
trainable_variables
regularization_losses
	variables
non_trainable_variables
metrics

layers
 layer_metrics
[__call__
\_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
?

kernel
bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
?
%axis
	gamma
beta
&trainable_variables
'regularization_losses
(	variables
)	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
?
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
?
.layer_with_weights-0
.layer-0
/trainable_variables
0regularization_losses
1	variables
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
trainable_variables
regularization_losses
	variables
4non_trainable_variables
5metrics

6layers
7layer_metrics
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:	?2	pi/kernel
:2pi/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
8layer_regularization_losses
trainable_variables
regularization_losses
	variables
9non_trainable_variables
:metrics

;layers
<layer_metrics
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:?2
dense/bias
(:&?2layer_normalization/gamma
':%?2layer_normalization/beta
": 
??2dense_1/kernel
:?2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
=layer_regularization_losses
!trainable_variables
"regularization_losses
#	variables
>non_trainable_variables
?metrics

@layers
Alayer_metrics
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Blayer_regularization_losses
&trainable_variables
'regularization_losses
(	variables
Cnon_trainable_variables
Dmetrics

Elayers
Flayer_metrics
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
*trainable_variables
+regularization_losses
,	variables
Hnon_trainable_variables
Imetrics

Jlayers
Klayer_metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?

kernel
bias
Ltrainable_variables
Mregularization_losses
N	variables
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
/trainable_variables
0regularization_losses
1	variables
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
	1

2
3"
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
 "
trackable_dict_wrapper
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
Ulayer_regularization_losses
Ltrainable_variables
Mregularization_losses
N	variables
Vnon_trainable_variables
Wmetrics

Xlayers
Ylayer_metrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
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
?2?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432202
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432271
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432340
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432409?
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
5__inference_private__mlp_actor_layer_call_fn_13432430
5__inference_private__mlp_actor_layer_call_fn_13432451
5__inference_private__mlp_actor_layer_call_fn_13432472
5__inference_private__mlp_actor_layer_call_fn_13432493?
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
#__inference__wrapped_model_13431570input_1"?
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
C__inference_actor_layer_call_and_return_conditional_losses_13432553
C__inference_actor_layer_call_and_return_conditional_losses_13432613
C__inference_actor_layer_call_and_return_conditional_losses_13431886
C__inference_actor_layer_call_and_return_conditional_losses_13431906?
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
(__inference_actor_layer_call_fn_13431769
(__inference_actor_layer_call_fn_13432630
(__inference_actor_layer_call_fn_13432647
(__inference_actor_layer_call_fn_13431866?
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
@__inference_pi_layer_call_and_return_conditional_losses_13432658?
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
%__inference_pi_layer_call_fn_13432667?
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
&__inference_signature_wrapper_13432133input_1"?
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
C__inference_dense_layer_call_and_return_conditional_losses_13432677?
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
(__inference_dense_layer_call_fn_13432686?
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
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_13432732?
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
6__inference_layer_normalization_layer_call_fn_13432741?
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
H__inference_activation_layer_call_and_return_conditional_losses_13432746?
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
-__inference_activation_layer_call_fn_13432751?
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
H__inference_sequential_layer_call_and_return_conditional_losses_13432762
H__inference_sequential_layer_call_and_return_conditional_losses_13432773
H__inference_sequential_layer_call_and_return_conditional_losses_13431657
H__inference_sequential_layer_call_and_return_conditional_losses_13431666?
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
-__inference_sequential_layer_call_fn_13431602
-__inference_sequential_layer_call_fn_13432782
-__inference_sequential_layer_call_fn_13432791
-__inference_sequential_layer_call_fn_13431648?
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
E__inference_dense_1_layer_call_and_return_conditional_losses_13432802?
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
*__inference_dense_1_layer_call_fn_13432811?
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
#__inference__wrapped_model_13431570q0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
H__inference_activation_layer_call_and_return_conditional_losses_13432746Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
-__inference_activation_layer_call_fn_13432751M0?-
&?#
!?
inputs??????????
? "????????????
C__inference_actor_layer_call_and_return_conditional_losses_13431886n<?9
2?/
%?"
dense_input?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_13431906n<?9
2?/
%?"
dense_input?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_13432553i7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_13432613i7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
(__inference_actor_layer_call_fn_13431769a<?9
2?/
%?"
dense_input?????????
p 

 
? "????????????
(__inference_actor_layer_call_fn_13431866a<?9
2?/
%?"
dense_input?????????
p

 
? "????????????
(__inference_actor_layer_call_fn_13432630\7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
(__inference_actor_layer_call_fn_13432647\7?4
-?*
 ?
inputs?????????
p

 
? "????????????
E__inference_dense_1_layer_call_and_return_conditional_losses_13432802^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_1_layer_call_fn_13432811Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_13432677]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? |
(__inference_dense_layer_call_fn_13432686P/?,
%?"
 ?
inputs?????????
? "????????????
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_13432732^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_layer_normalization_layer_call_fn_13432741Q0?-
&?#
!?
inputs??????????
? "????????????
@__inference_pi_layer_call_and_return_conditional_losses_13432658]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_pi_layer_call_fn_13432667P0?-
&?#
!?
inputs??????????
? "???????????
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432202a.?+
$?!
?
x?????????
p 
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432271a.?+
$?!
?
x?????????
p
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432340g4?1
*?'
!?
input_1?????????
p 
? "%?"
?
0?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_13432409g4?1
*?'
!?
input_1?????????
p
? "%?"
?
0?????????
? ?
5__inference_private__mlp_actor_layer_call_fn_13432430Z4?1
*?'
!?
input_1?????????
p 
? "???????????
5__inference_private__mlp_actor_layer_call_fn_13432451T.?+
$?!
?
x?????????
p 
? "???????????
5__inference_private__mlp_actor_layer_call_fn_13432472T.?+
$?!
?
x?????????
p
? "???????????
5__inference_private__mlp_actor_layer_call_fn_13432493Z4?1
*?'
!?
input_1?????????
p
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_13431657m??<
5?2
(?%
dense_1_input??????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_13431666m??<
5?2
(?%
dense_1_input??????????
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_13432762f8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_13432773f8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
-__inference_sequential_layer_call_fn_13431602`??<
5?2
(?%
dense_1_input??????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_13431648`??<
5?2
(?%
dense_1_input??????????
p

 
? "????????????
-__inference_sequential_layer_call_fn_13432782Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_13432791Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
&__inference_signature_wrapper_13432133|;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????