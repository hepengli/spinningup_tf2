	
??
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
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
private__mlp_actor/mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name private__mlp_actor/mean/kernel
?
2private__mlp_actor/mean/kernel/Read/ReadVariableOpReadVariableOpprivate__mlp_actor/mean/kernel*
_output_shapes
:	?*
dtype0
?
private__mlp_actor/mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameprivate__mlp_actor/mean/bias
?
0private__mlp_actor/mean/bias/Read/ReadVariableOpReadVariableOpprivate__mlp_actor/mean/bias*
_output_shapes
:*
dtype0
?
%private__mlp_actor/log_std_dev/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%private__mlp_actor/log_std_dev/kernel
?
9private__mlp_actor/log_std_dev/kernel/Read/ReadVariableOpReadVariableOp%private__mlp_actor/log_std_dev/kernel*
_output_shapes
:	?*
dtype0
?
#private__mlp_actor/log_std_dev/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#private__mlp_actor/log_std_dev/bias
?
7private__mlp_actor/log_std_dev/bias/Read/ReadVariableOpReadVariableOp#private__mlp_actor/log_std_dev/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	o?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	o?*
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
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	_body
_mu
_log_std
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
?

layers
 metrics
!layer_metrics
regularization_losses
"non_trainable_variables
trainable_variables
#layer_regularization_losses
	variables
 
h

kernel
bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
 

0
1
2
3

0
1
2
3
?

,layers
-metrics
.layer_metrics
regularization_losses
/non_trainable_variables
trainable_variables
0layer_regularization_losses
	variables
YW
VARIABLE_VALUEprivate__mlp_actor/mean/kernel%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEprivate__mlp_actor/mean/bias#_mu/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

1layers
2metrics
3layer_metrics
regularization_losses
4non_trainable_variables
trainable_variables
5layer_regularization_losses
	variables
ec
VARIABLE_VALUE%private__mlp_actor/log_std_dev/kernel*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE#private__mlp_actor/log_std_dev/bias(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

6layers
7metrics
8layer_metrics
regularization_losses
9non_trainable_variables
trainable_variables
:layer_regularization_losses
	variables
RP
VARIABLE_VALUEdense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 
 
 

0
1

0
1
?

;layers
<metrics
=layer_metrics
$regularization_losses
>non_trainable_variables
%trainable_variables
?layer_regularization_losses
&	variables
 

0
1

0
1
?

@layers
Ametrics
Blayer_metrics
(regularization_losses
Cnon_trainable_variables
)trainable_variables
Dlayer_regularization_losses
*	variables

	0

1
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
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????o*
dtype0*
shape:?????????o
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasprivate__mlp_actor/mean/kernelprivate__mlp_actor/mean/bias%private__mlp_actor/log_std_dev/kernel#private__mlp_actor/log_std_dev/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_79437408
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2private__mlp_actor/mean/kernel/Read/ReadVariableOp0private__mlp_actor/mean/bias/Read/ReadVariableOp9private__mlp_actor/log_std_dev/kernel/Read/ReadVariableOp7private__mlp_actor/log_std_dev/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_79438071
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameprivate__mlp_actor/mean/kernelprivate__mlp_actor/mean/bias%private__mlp_actor/log_std_dev/kernel#private__mlp_actor/log_std_dev/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
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
$__inference__traced_restore_79438105??
?
?
'__inference_mean_layer_call_fn_79437963

inputs
unknown:	?
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
GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_794369792
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_actor_layer_call_fn_79437905
dense_input
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794368412
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
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?

?
I__inference_log_std_dev_layer_call_and_return_conditional_losses_79436995

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_79438002

inputs
unknown:	o?
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
C__inference_dense_layer_call_and_return_conditional_losses_794368172
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
:?????????o: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_79437993

inputs1
matmul_readvariableop_resource:	o?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	o?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????o: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?<
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437191
x!
actor_79437123:	o?
actor_79437125:	?"
actor_79437127:
??
actor_79437129:	? 
mean_79437132:	?
mean_79437134:'
log_std_dev_79437137:	?"
log_std_dev_79437139:
identity

identity_1

identity_2??actor/StatefulPartitionedCall?#log_std_dev/StatefulPartitionedCall?mean/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_79437123actor_79437125actor_79437127actor_79437129*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794369012
actor/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0mean_79437132mean_79437134*
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
GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_794369792
mean/StatefulPartitionedCall?
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0log_std_dev_79437137log_std_dev_79437139*
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
GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_794369952%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Expc
ShapeShape%mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulu
addAddV2%mean/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
adds
subSubadd:z:0%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3m
TanhTanh%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp^actor/StatefulPartitionedCall$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?
?
C__inference_actor_layer_call_and_return_conditional_losses_79437892
dense_input7
$dense_matmul_readvariableop_resource:	o?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
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
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?
?
(__inference_actor_layer_call_fn_79437931

inputs
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794369012
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
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
.__inference_log_std_dev_layer_call_fn_79437982

inputs
unknown:	?
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
GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_794369952
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
5__inference_private__mlp_actor_layer_call_fn_79437795
x
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_794371912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?'
?
$__inference__traced_restore_79438105
file_prefixB
/assignvariableop_private__mlp_actor_mean_kernel:	?=
/assignvariableop_1_private__mlp_actor_mean_bias:K
8assignvariableop_2_private__mlp_actor_log_std_dev_kernel:	?D
6assignvariableop_3_private__mlp_actor_log_std_dev_bias:2
assignvariableop_4_dense_kernel:	o?,
assignvariableop_5_dense_bias:	?5
!assignvariableop_6_dense_1_kernel:
??.
assignvariableop_7_dense_1_bias:	?

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp/assignvariableop_private__mlp_actor_mean_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_private__mlp_actor_mean_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp8assignvariableop_2_private__mlp_actor_log_std_dev_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_private__mlp_actor_log_std_dev_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
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
?
?
*__inference_dense_1_layer_call_fn_79438022

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
E__inference_dense_1_layer_call_and_return_conditional_losses_794368342
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
?P
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437564
x=
*actor_dense_matmul_readvariableop_resource:	o?:
+actor_dense_biasadd_readvariableop_resource:	?@
,actor_dense_1_matmul_readvariableop_resource:
??<
-actor_dense_1_biasadd_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?$actor/dense_1/BiasAdd/ReadVariableOp?#actor/dense_1/MatMul/ReadVariableOp?"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense/Relu?
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#actor/dense_1/MatMul/ReadVariableOp?
actor/dense_1/MatMulMatMulactor/dense/Relu:activations:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/MatMul?
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOp?
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/BiasAdd?
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/Relu?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul actor/dense_1/Relu:activations:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul actor/dense_1/Relu:activations:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
ExpS
ShapeShapemean/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2L
$actor/dense_1/BiasAdd/ReadVariableOp$actor/dense_1/BiasAdd/ReadVariableOp2J
#actor/dense_1/MatMul/ReadVariableOp#actor/dense_1/MatMul/ReadVariableOp2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????o

_user_specified_namex
?

?
B__inference_mean_layer_call_and_return_conditional_losses_79436979

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
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
?
5__inference_private__mlp_actor_layer_call_fn_79437820
input_1
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_794371912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?Q
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437720
input_1=
*actor_dense_matmul_readvariableop_resource:	o?:
+actor_dense_biasadd_readvariableop_resource:	?@
,actor_dense_1_matmul_readvariableop_resource:
??<
-actor_dense_1_biasadd_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?$actor/dense_1/BiasAdd/ReadVariableOp?#actor/dense_1/MatMul/ReadVariableOp?"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense/Relu?
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#actor/dense_1/MatMul/ReadVariableOp?
actor/dense_1/MatMulMatMulactor/dense/Relu:activations:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/MatMul?
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOp?
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/BiasAdd?
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/Relu?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul actor/dense_1/Relu:activations:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul actor/dense_1/Relu:activations:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
ExpS
ShapeShapemean/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2L
$actor/dense_1/BiasAdd/ReadVariableOp$actor/dense_1/BiasAdd/ReadVariableOp2J
#actor/dense_1/MatMul/ReadVariableOp#actor/dense_1/MatMul/ReadVariableOp2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?P
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437486
x=
*actor_dense_matmul_readvariableop_resource:	o?:
+actor_dense_biasadd_readvariableop_resource:	?@
,actor_dense_1_matmul_readvariableop_resource:
??<
-actor_dense_1_biasadd_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?$actor/dense_1/BiasAdd/ReadVariableOp?#actor/dense_1/MatMul/ReadVariableOp?"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense/Relu?
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#actor/dense_1/MatMul/ReadVariableOp?
actor/dense_1/MatMulMatMulactor/dense/Relu:activations:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/MatMul?
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOp?
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/BiasAdd?
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/Relu?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul actor/dense_1/Relu:activations:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul actor/dense_1/Relu:activations:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
ExpS
ShapeShapemean/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2L
$actor/dense_1/BiasAdd/ReadVariableOp$actor/dense_1/BiasAdd/ReadVariableOp2J
#actor/dense_1/MatMul/ReadVariableOp#actor/dense_1/MatMul/ReadVariableOp2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????o

_user_specified_namex
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_79438013

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
?
C__inference_actor_layer_call_and_return_conditional_losses_79436901

inputs!
dense_79436890:	o?
dense_79436892:	?$
dense_1_79436895:
??
dense_1_79436897:	?
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_79436890dense_79436892*
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
C__inference_dense_layer_call_and_return_conditional_losses_794368172
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_79436895dense_1_79436897*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_794368342!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
(__inference_actor_layer_call_fn_79437918

inputs
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794368412
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
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?

?
I__inference_log_std_dev_layer_call_and_return_conditional_losses_79437973

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_79437856

inputs7
$dense_matmul_readvariableop_resource:	o?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
(__inference_actor_layer_call_fn_79437944
dense_input
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794369012
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
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?u
?	
#__inference__wrapped_model_79436799
input_1P
=private__mlp_actor_actor_dense_matmul_readvariableop_resource:	o?M
>private__mlp_actor_actor_dense_biasadd_readvariableop_resource:	?S
?private__mlp_actor_actor_dense_1_matmul_readvariableop_resource:
??O
@private__mlp_actor_actor_dense_1_biasadd_readvariableop_resource:	?I
6private__mlp_actor_mean_matmul_readvariableop_resource:	?E
7private__mlp_actor_mean_biasadd_readvariableop_resource:P
=private__mlp_actor_log_std_dev_matmul_readvariableop_resource:	?L
>private__mlp_actor_log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2??5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp?4private__mlp_actor/actor/dense/MatMul/ReadVariableOp?7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp?6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp?5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp?4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp?.private__mlp_actor/mean/BiasAdd/ReadVariableOp?-private__mlp_actor/mean/MatMul/ReadVariableOp?
4private__mlp_actor/actor/dense/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_actor_dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
#private__mlp_actor/actor/dense/ReluRelu/private__mlp_actor/actor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2%
#private__mlp_actor/actor/dense/Relu?
6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOpReadVariableOp?private__mlp_actor_actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp?
'private__mlp_actor/actor/dense_1/MatMulMatMul1private__mlp_actor/actor/dense/Relu:activations:0>private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'private__mlp_actor/actor/dense_1/MatMul?
7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp@private__mlp_actor_actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp?
(private__mlp_actor/actor/dense_1/BiasAddBiasAdd1private__mlp_actor/actor/dense_1/MatMul:product:0?private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(private__mlp_actor/actor/dense_1/BiasAdd?
%private__mlp_actor/actor/dense_1/ReluRelu1private__mlp_actor/actor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%private__mlp_actor/actor/dense_1/Relu?
-private__mlp_actor/mean/MatMul/ReadVariableOpReadVariableOp6private__mlp_actor_mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-private__mlp_actor/mean/MatMul/ReadVariableOp?
private__mlp_actor/mean/MatMulMatMul3private__mlp_actor/actor/dense_1/Relu:activations:05private__mlp_actor/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
private__mlp_actor/mean/MatMul?
.private__mlp_actor/mean/BiasAdd/ReadVariableOpReadVariableOp7private__mlp_actor_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.private__mlp_actor/mean/BiasAdd/ReadVariableOp?
private__mlp_actor/mean/BiasAddBiasAdd(private__mlp_actor/mean/MatMul:product:06private__mlp_actor/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
private__mlp_actor/mean/BiasAdd?
4private__mlp_actor/log_std_dev/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp?
%private__mlp_actor/log_std_dev/MatMulMatMul3private__mlp_actor/actor/dense_1/Relu:activations:0<private__mlp_actor/log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%private__mlp_actor/log_std_dev/MatMul?
5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOpReadVariableOp>private__mlp_actor_log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp?
&private__mlp_actor/log_std_dev/BiasAddBiasAdd/private__mlp_actor/log_std_dev/MatMul:product:0=private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&private__mlp_actor/log_std_dev/BiasAdd?
*private__mlp_actor/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2,
*private__mlp_actor/clip_by_value/Minimum/y?
(private__mlp_actor/clip_by_value/MinimumMinimum/private__mlp_actor/log_std_dev/BiasAdd:output:03private__mlp_actor/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(private__mlp_actor/clip_by_value/Minimum?
"private__mlp_actor/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"private__mlp_actor/clip_by_value/y?
 private__mlp_actor/clip_by_valueMaximum,private__mlp_actor/clip_by_value/Minimum:z:0+private__mlp_actor/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 private__mlp_actor/clip_by_value?
private__mlp_actor/ExpExp$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/Exp?
private__mlp_actor/ShapeShape(private__mlp_actor/mean/BiasAdd:output:0*
T0*
_output_shapes
:2
private__mlp_actor/Shape?
%private__mlp_actor/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%private__mlp_actor/random_normal/mean?
'private__mlp_actor/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'private__mlp_actor/random_normal/stddev?
5private__mlp_actor/random_normal/RandomStandardNormalRandomStandardNormal!private__mlp_actor/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed27
5private__mlp_actor/random_normal/RandomStandardNormal?
$private__mlp_actor/random_normal/mulMul>private__mlp_actor/random_normal/RandomStandardNormal:output:00private__mlp_actor/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2&
$private__mlp_actor/random_normal/mul?
 private__mlp_actor/random_normalAddV2(private__mlp_actor/random_normal/mul:z:0.private__mlp_actor/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2"
 private__mlp_actor/random_normal?
private__mlp_actor/mulMul$private__mlp_actor/random_normal:z:0private__mlp_actor/Exp:y:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul?
private__mlp_actor/addAddV2(private__mlp_actor/mean/BiasAdd:output:0private__mlp_actor/mul:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/add?
private__mlp_actor/subSubprivate__mlp_actor/add:z:0(private__mlp_actor/mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/sub?
private__mlp_actor/Exp_1Exp$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/Exp_1}
private__mlp_actor/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
private__mlp_actor/add_1/y?
private__mlp_actor/add_1AddV2private__mlp_actor/Exp_1:y:0#private__mlp_actor/add_1/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/add_1?
private__mlp_actor/truedivRealDivprivate__mlp_actor/sub:z:0private__mlp_actor/add_1:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/truedivy
private__mlp_actor/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/pow/y?
private__mlp_actor/powPowprivate__mlp_actor/truediv:z:0!private__mlp_actor/pow/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/pow}
private__mlp_actor/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/mul_1/x?
private__mlp_actor/mul_1Mul#private__mlp_actor/mul_1/x:output:0$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_1?
private__mlp_actor/add_2AddV2private__mlp_actor/pow:z:0private__mlp_actor/mul_1:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/add_2}
private__mlp_actor/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2
private__mlp_actor/add_3/y?
private__mlp_actor/add_3AddV2private__mlp_actor/add_2:z:0#private__mlp_actor/add_3/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/add_3}
private__mlp_actor/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
private__mlp_actor/mul_2/x?
private__mlp_actor/mul_2Mul#private__mlp_actor/mul_2/x:output:0private__mlp_actor/add_3:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_2?
(private__mlp_actor/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(private__mlp_actor/Sum/reduction_indices?
private__mlp_actor/SumSumprivate__mlp_actor/mul_2:z:01private__mlp_actor/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
private__mlp_actor/Sum}
private__mlp_actor/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2
private__mlp_actor/sub_1/x?
private__mlp_actor/sub_1Sub#private__mlp_actor/sub_1/x:output:0private__mlp_actor/add:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/sub_1}
private__mlp_actor/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
private__mlp_actor/mul_3/x?
private__mlp_actor/mul_3Mul#private__mlp_actor/mul_3/x:output:0private__mlp_actor/add:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_3?
private__mlp_actor/SoftplusSoftplusprivate__mlp_actor/mul_3:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/Softplus?
private__mlp_actor/sub_2Subprivate__mlp_actor/sub_1:z:0)private__mlp_actor/Softplus:activations:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/sub_2}
private__mlp_actor/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/mul_4/x?
private__mlp_actor/mul_4Mul#private__mlp_actor/mul_4/x:output:0private__mlp_actor/sub_2:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_4?
*private__mlp_actor/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*private__mlp_actor/Sum_1/reduction_indices?
private__mlp_actor/Sum_1Sumprivate__mlp_actor/mul_4:z:03private__mlp_actor/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
private__mlp_actor/Sum_1?
private__mlp_actor/sub_3Subprivate__mlp_actor/Sum:output:0!private__mlp_actor/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
private__mlp_actor/sub_3?
private__mlp_actor/TanhTanh(private__mlp_actor/mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/Tanh?
private__mlp_actor/Tanh_1Tanhprivate__mlp_actor/add:z:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/Tanh_1}
private__mlp_actor/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
private__mlp_actor/mul_5/y?
private__mlp_actor/mul_5Mulprivate__mlp_actor/Tanh:y:0#private__mlp_actor/mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_5}
private__mlp_actor/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
private__mlp_actor/mul_6/y?
private__mlp_actor/mul_6Mulprivate__mlp_actor/Tanh_1:y:0#private__mlp_actor/mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
private__mlp_actor/mul_6w
IdentityIdentityprivate__mlp_actor/mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity{

Identity_1Identityprivate__mlp_actor/mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1w

Identity_2Identityprivate__mlp_actor/sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp6^private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5^private__mlp_actor/actor/dense/MatMul/ReadVariableOp8^private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp7^private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp6^private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp5^private__mlp_actor/log_std_dev/MatMul/ReadVariableOp/^private__mlp_actor/mean/BiasAdd/ReadVariableOp.^private__mlp_actor/mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2n
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp2l
4private__mlp_actor/actor/dense/MatMul/ReadVariableOp4private__mlp_actor/actor/dense/MatMul/ReadVariableOp2r
7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp2p
6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp2n
5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp2l
4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp2`
.private__mlp_actor/mean/BiasAdd/ReadVariableOp.private__mlp_actor/mean/BiasAdd/ReadVariableOp2^
-private__mlp_actor/mean/MatMul/ReadVariableOp-private__mlp_actor/mean/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?Q
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437642
input_1=
*actor_dense_matmul_readvariableop_resource:	o?:
+actor_dense_biasadd_readvariableop_resource:	?@
,actor_dense_1_matmul_readvariableop_resource:
??<
-actor_dense_1_biasadd_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?$actor/dense_1/BiasAdd/ReadVariableOp?#actor/dense_1/MatMul/ReadVariableOp?"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense/Relu?
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#actor/dense_1/MatMul/ReadVariableOp?
actor/dense_1/MatMulMatMulactor/dense/Relu:activations:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/MatMul?
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOp?
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/BiasAdd?
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense_1/Relu?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul actor/dense_1/Relu:activations:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul actor/dense_1/Relu:activations:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
ExpS
ShapeShapemean/BiasAdd:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2L
$actor/dense_1/BiasAdd/ReadVariableOp$actor/dense_1/BiasAdd/ReadVariableOp2J
#actor/dense_1/MatMul/ReadVariableOp#actor/dense_1/MatMul/ReadVariableOp2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
&__inference_signature_wrapper_79437408
input_1
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_794367992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
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
5__inference_private__mlp_actor_layer_call_fn_79437745
input_1
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_794370502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????o
!
_user_specified_name	input_1
?
?
C__inference_actor_layer_call_and_return_conditional_losses_79437874
dense_input7
$dense_matmul_readvariableop_resource:	o?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
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
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????o
%
_user_specified_namedense_input
?
?
C__inference_dense_layer_call_and_return_conditional_losses_79436817

inputs1
matmul_readvariableop_resource:	o?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	o?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????o: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
!__inference__traced_save_79438071
file_prefix=
9savev2_private__mlp_actor_mean_kernel_read_readvariableop;
7savev2_private__mlp_actor_mean_bias_read_readvariableopD
@savev2_private__mlp_actor_log_std_dev_kernel_read_readvariableopB
>savev2_private__mlp_actor_log_std_dev_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
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
value?B?	B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_private__mlp_actor_mean_kernel_read_readvariableop7savev2_private__mlp_actor_mean_bias_read_readvariableop@savev2_private__mlp_actor_log_std_dev_kernel_read_readvariableop>savev2_private__mlp_actor_log_std_dev_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*^
_input_shapesM
K: :	?::	?::	o?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	o?:!
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
?
?
C__inference_actor_layer_call_and_return_conditional_losses_79437838

inputs7
$dense_matmul_readvariableop_resource:	o?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	o?*
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
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_79436841

inputs!
dense_79436818:	o?
dense_79436820:	?$
dense_1_79436835:
??
dense_1_79436837:	?
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_79436818dense_79436820*
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
C__inference_dense_layer_call_and_return_conditional_losses_794368172
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_79436835dense_1_79436837*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_794368342!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????o: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????o
 
_user_specified_nameinputs
?

?
B__inference_mean_layer_call_and_return_conditional_losses_79437954

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437050
x!
actor_79436960:	o?
actor_79436962:	?"
actor_79436964:
??
actor_79436966:	? 
mean_79436980:	?
mean_79436982:'
log_std_dev_79436996:	?"
log_std_dev_79436998:
identity

identity_1

identity_2??actor/StatefulPartitionedCall?#log_std_dev/StatefulPartitionedCall?mean/StatefulPartitionedCall?
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_79436960actor_79436962actor_79436964actor_79436966*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_794368412
actor/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0mean_79436980mean_79436982*
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
GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_794369792
mean/StatefulPartitionedCall?
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0log_std_dev_79436996log_std_dev_79436998*
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
GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_794369952%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Expc
ShapeShape%mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulu
addAddV2%mean/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
adds
subSubadd:z:0%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:?????????2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:?????????2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y`
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xl
mul_1Mulmul_1/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:?????????2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:?????????2
mul_2p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_2:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
sub_1/xb
sub_1Subsub_1/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:?????????2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:?????????2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:?????????2
sub_2W
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_4/xd
mul_4Mulmul_4/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:?????????2
mul_4t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesp
Sum_1Sum	mul_4:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:?????????2
sub_3m
TanhTanh%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:?????????2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:?????????2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2?
NoOpNoOp^actor/StatefulPartitionedCall$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????o: : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:J F
'
_output_shapes
:?????????o

_user_specified_namex
?
?
5__inference_private__mlp_actor_layer_call_fn_79437770
x
unknown:	o?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_794370502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:?????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
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
E__inference_dense_1_layer_call_and_return_conditional_losses_79436834

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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????o<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????8
output_3,
StatefulPartitionedCall:2?????????tensorflow/serving/predict:?o
?
	_body
_mu
_log_std
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*E&call_and_return_all_conditional_losses
F_default_save_signature
G__call__"
_tf_keras_model
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_sequential
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"
_tf_keras_layer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?

layers
 metrics
!layer_metrics
regularization_losses
"non_trainable_variables
trainable_variables
#layer_regularization_losses
	variables
G__call__
F_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
?

kernel
bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
*O&call_and_return_all_conditional_losses
P__call__"
_tf_keras_layer
?

kernel
bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"
_tf_keras_layer
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?

,layers
-metrics
.layer_metrics
regularization_losses
/non_trainable_variables
trainable_variables
0layer_regularization_losses
	variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
1:/	?2private__mlp_actor/mean/kernel
*:(2private__mlp_actor/mean/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

1layers
2metrics
3layer_metrics
regularization_losses
4non_trainable_variables
trainable_variables
5layer_regularization_losses
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
8:6	?2%private__mlp_actor/log_std_dev/kernel
1:/2#private__mlp_actor/log_std_dev/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

6layers
7metrics
8layer_metrics
regularization_losses
9non_trainable_variables
trainable_variables
:layer_regularization_losses
	variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:	o?2dense/kernel
:?2
dense/bias
": 
??2dense_1/kernel
:?2dense_1/bias
5
0
1
2"
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

;layers
<metrics
=layer_metrics
$regularization_losses
>non_trainable_variables
%trainable_variables
?layer_regularization_losses
&	variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

@layers
Ametrics
Blayer_metrics
(regularization_losses
Cnon_trainable_variables
)trainable_variables
Dlayer_regularization_losses
*	variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
.
	0

1"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437486
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437564
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437642
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437720?
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
#__inference__wrapped_model_79436799input_1"?
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
5__inference_private__mlp_actor_layer_call_fn_79437745
5__inference_private__mlp_actor_layer_call_fn_79437770
5__inference_private__mlp_actor_layer_call_fn_79437795
5__inference_private__mlp_actor_layer_call_fn_79437820?
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
C__inference_actor_layer_call_and_return_conditional_losses_79437838
C__inference_actor_layer_call_and_return_conditional_losses_79437856
C__inference_actor_layer_call_and_return_conditional_losses_79437874
C__inference_actor_layer_call_and_return_conditional_losses_79437892?
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
(__inference_actor_layer_call_fn_79437905
(__inference_actor_layer_call_fn_79437918
(__inference_actor_layer_call_fn_79437931
(__inference_actor_layer_call_fn_79437944?
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
B__inference_mean_layer_call_and_return_conditional_losses_79437954?
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
'__inference_mean_layer_call_fn_79437963?
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
I__inference_log_std_dev_layer_call_and_return_conditional_losses_79437973?
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
.__inference_log_std_dev_layer_call_fn_79437982?
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
&__inference_signature_wrapper_79437408input_1"?
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
C__inference_dense_layer_call_and_return_conditional_losses_79437993?
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
(__inference_dense_layer_call_fn_79438002?
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
E__inference_dense_1_layer_call_and_return_conditional_losses_79438013?
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
*__inference_dense_1_layer_call_fn_79438022?
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
#__inference__wrapped_model_79436799?0?-
&?#
!?
input_1?????????o
? "???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
*
output_3?
output_3??????????
C__inference_actor_layer_call_and_return_conditional_losses_79437838g7?4
-?*
 ?
inputs?????????o
p 

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_79437856g7?4
-?*
 ?
inputs?????????o
p

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_79437874l<?9
2?/
%?"
dense_input?????????o
p 

 
? "&?#
?
0??????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_79437892l<?9
2?/
%?"
dense_input?????????o
p

 
? "&?#
?
0??????????
? ?
(__inference_actor_layer_call_fn_79437905_<?9
2?/
%?"
dense_input?????????o
p 

 
? "????????????
(__inference_actor_layer_call_fn_79437918Z7?4
-?*
 ?
inputs?????????o
p 

 
? "????????????
(__inference_actor_layer_call_fn_79437931Z7?4
-?*
 ?
inputs?????????o
p

 
? "????????????
(__inference_actor_layer_call_fn_79437944_<?9
2?/
%?"
dense_input?????????o
p

 
? "????????????
E__inference_dense_1_layer_call_and_return_conditional_losses_79438013^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_1_layer_call_fn_79438022Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_79437993]/?,
%?"
 ?
inputs?????????o
? "&?#
?
0??????????
? |
(__inference_dense_layer_call_fn_79438002P/?,
%?"
 ?
inputs?????????o
? "????????????
I__inference_log_std_dev_layer_call_and_return_conditional_losses_79437973]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
.__inference_log_std_dev_layer_call_fn_79437982P0?-
&?#
!?
inputs??????????
? "???????????
B__inference_mean_layer_call_and_return_conditional_losses_79437954]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_mean_layer_call_fn_79437963P0?-
&?#
!?
inputs??????????
? "???????????
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437486?.?+
$?!
?
x?????????o
p 
? "f?c
\?Y
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437564?.?+
$?!
?
x?????????o
p
? "f?c
\?Y
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437642?4?1
*?'
!?
input_1?????????o
p 
? "f?c
\?Y
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_79437720?4?1
*?'
!?
input_1?????????o
p
? "f?c
\?Y
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
5__inference_private__mlp_actor_layer_call_fn_79437745?4?1
*?'
!?
input_1?????????o
p 
? "V?S
?
0?????????
?
1?????????
?
2??????????
5__inference_private__mlp_actor_layer_call_fn_79437770?.?+
$?!
?
x?????????o
p 
? "V?S
?
0?????????
?
1?????????
?
2??????????
5__inference_private__mlp_actor_layer_call_fn_79437795?.?+
$?!
?
x?????????o
p
? "V?S
?
0?????????
?
1?????????
?
2??????????
5__inference_private__mlp_actor_layer_call_fn_79437820?4?1
*?'
!?
input_1?????????o
p
? "V?S
?
0?????????
?
1?????????
?
2??????????
&__inference_signature_wrapper_79437408?;?8
? 
1?.
,
input_1!?
input_1?????????o"???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
*
output_3?
output_3?????????