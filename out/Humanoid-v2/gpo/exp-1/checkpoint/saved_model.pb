??
??
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
r
log_std_devVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelog_std_dev
k
log_std_dev/Read/ReadVariableOpReadVariableOplog_std_dev*
_output_shapes

:*
dtype0
s
mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namemean/kernel
l
mean/kernel/Read/ReadVariableOpReadVariableOpmean/kernel*
_output_shapes
:	?*
dtype0
j
	mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	mean/bias
c
mean/bias/Read/ReadVariableOpReadVariableOp	mean/bias*
_output_shapes
:*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
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
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
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
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:?*
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
	_body
_mu
_log_std
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
DB
VARIABLE_VALUElog_std_dev#_log_std/.ATTRIBUTES/VARIABLE_VALUE
^
0
1
2
3
4
5
6
7
 8
!9
10
11
12
?
0
1
"2
#3
4
5
6
7
$8
%9
10
11
 12
!13
&14
'15
16
17
18
 
?
(metrics
trainable_variables
	variables
regularization_losses
)non_trainable_variables

*layers
+layer_metrics
,layer_regularization_losses
 
?
-axis
	gamma
beta
"moving_mean
#moving_variance
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

kernel
bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
?
6axis
	gamma
beta
$moving_mean
%moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

kernel
bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?
?axis
	 gamma
!beta
&moving_mean
'moving_variance
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
F
0
1
2
3
4
5
6
7
 8
!9
v
0
1
"2
#3
4
5
6
7
$8
%9
10
11
 12
!13
&14
'15
 
?
Dmetrics
trainable_variables
	variables
regularization_losses
Enon_trainable_variables

Flayers
Glayer_metrics
Hlayer_regularization_losses
FD
VARIABLE_VALUEmean/kernel%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUE	mean/bias#_mu/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses

Klayers
Llayer_metrics
Mmetrics
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_1/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_1/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_2/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_2/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_1/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_1/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
*
"0
#1
$2
%3
&4
'5

0
1
 
 
 

0
1

0
1
"2
#3
 
?
.trainable_variables
/	variables
0regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses

Players
Qlayer_metrics
Rmetrics

0
1

0
1
 
?
2trainable_variables
3	variables
4regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
Vlayer_metrics
Wmetrics
 

0
1

0
1
$2
%3
 
?
7trainable_variables
8	variables
9regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
[layer_metrics
\metrics

0
1

0
1
 
?
;trainable_variables
<	variables
=regularization_losses
]non_trainable_variables
^layer_regularization_losses

_layers
`layer_metrics
ametrics
 

 0
!1

 0
!1
&2
'3
 
?
@trainable_variables
A	variables
Bregularization_losses
bnon_trainable_variables
clayer_regularization_losses

dlayers
elayer_metrics
fmetrics
 
*
"0
#1
$2
%3
&4
'5
#
	0

1
2
3
4
 
 
 
 
 
 
 

"0
#1
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
$0
%1
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
&0
'1
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammadense/kernel
dense/bias!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_1/betabatch_normalization_1/gammadense_1/kerneldense_1/bias!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_2/betabatch_normalization_2/gammamean/kernel	mean/biaslog_std_dev*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_33438085
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamelog_std_dev/Read/ReadVariableOpmean/kernel/Read/ReadVariableOpmean/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpConst* 
Tin
2*
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
!__inference__traced_save_33439293
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelog_std_devmean/kernel	mean/biasbatch_normalization/gammabatch_normalization/betadense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/betadense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance*
Tin
2*
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
$__inference__traced_restore_33439360??
?
?
C__inference_dense_layer_call_and_return_conditional_losses_33437308

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
?)
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33437041

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439112

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
މ
?
__inference_call_4888391

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_33438945

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368192
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_logprob_4888610
observations	
value'
continuous_actor_4888553:	?'
continuous_actor_4888555:	?'
continuous_actor_4888557:	?'
continuous_actor_4888559:	?,
continuous_actor_4888561:
??'
continuous_actor_4888563:	?'
continuous_actor_4888565:	?'
continuous_actor_4888567:	?'
continuous_actor_4888569:	?'
continuous_actor_4888571:	?,
continuous_actor_4888573:
??'
continuous_actor_4888575:	?'
continuous_actor_4888577:	?'
continuous_actor_4888579:	?'
continuous_actor_4888581:	?'
continuous_actor_4888583:	?+
continuous_actor_4888585:	?&
continuous_actor_4888587:*
continuous_actor_4888589:
identity??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallobservationscontinuous_actor_4888553continuous_actor_4888555continuous_actor_4888557continuous_actor_4888559continuous_actor_4888561continuous_actor_4888563continuous_actor_4888565continuous_actor_4888567continuous_actor_4888569continuous_actor_4888571continuous_actor_4888573continuous_actor_4888575continuous_actor_4888577continuous_actor_4888579continuous_actor_4888581continuous_actor_4888583continuous_actor_4888585continuous_actor_4888587continuous_actor_4888589*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:	?:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_48882012*
(continuous_actor/StatefulPartitionedCally
subSubvalue1continuous_actor/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:
?2
subm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
ExpS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/yU
addAddV2Exp:y:0add/y:output:0*
T0*
_output_shapes

:2
add]
truedivRealDivsub:z:0add:z:0*
T0*#
_output_shapes
:
?2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowtruediv:z:0pow/y:output:0*
T0*#
_output_shapes
:
?2
powS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/x}
mulMulmul/x:output:01continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
mulW
add_1AddV2pow:z:0mul:z:0*
T0*#
_output_shapes
:
?2
add_1W
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_2/yb
add_2AddV2	add_1:z:0add_2/y:output:0*
T0*#
_output_shapes
:
?2
add_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x`
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*#
_output_shapes
:
?2
mul_1{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicesj
MeanMean	mul_1:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:	
?2
Mean`
IdentityIdentityMean:output:0^NoOp*
T0*
_output_shapes
:	
?2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:
??:
?: : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:N J
 
_output_shapes
:
??
&
_user_specified_nameobservations:JF
#
_output_shapes
:
?

_user_specified_namevalue
?%
?
H__inference_sequential_layer_call_and_return_conditional_losses_33437598
batch_normalization_input+
batch_normalization_33437560:	?+
batch_normalization_33437562:	?+
batch_normalization_33437564:	?+
batch_normalization_33437566:	?"
dense_33437569:
??
dense_33437571:	?-
batch_normalization_1_33437574:	?-
batch_normalization_1_33437576:	?-
batch_normalization_1_33437578:	?-
batch_normalization_1_33437580:	?$
dense_1_33437583:
??
dense_1_33437585:	?-
batch_normalization_2_33437588:	?-
batch_normalization_2_33437590:	?-
batch_normalization_2_33437592:	?-
batch_normalization_2_33437594:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_33437560batch_normalization_33437562batch_normalization_33437564batch_normalization_33437566*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368192-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_33437569dense_33437571*
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
C__inference_dense_layer_call_and_return_conditional_losses_334373082
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_33437574batch_normalization_1_33437576batch_normalization_1_33437578batch_normalization_1_33437580*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334369812/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_33437583dense_1_33437585*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_334373342!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_33437588batch_normalization_2_33437590batch_normalization_2_33437592batch_normalization_2_33437594*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334371432/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:c _
(
_output_shapes
:??????????
3
_user_specified_namebatch_normalization_input
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33438978

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
__inference_action_4888120
observations'
continuous_actor_4888065:	?'
continuous_actor_4888067:	?'
continuous_actor_4888069:	?'
continuous_actor_4888071:	?,
continuous_actor_4888073:
??'
continuous_actor_4888075:	?'
continuous_actor_4888077:	?'
continuous_actor_4888079:	?'
continuous_actor_4888081:	?'
continuous_actor_4888083:	?,
continuous_actor_4888085:
??'
continuous_actor_4888087:	?'
continuous_actor_4888089:	?'
continuous_actor_4888091:	?'
continuous_actor_4888093:	?'
continuous_actor_4888095:	?+
continuous_actor_4888097:	?&
continuous_actor_4888099:*
continuous_actor_4888101:
identity??(continuous_actor/StatefulPartitionedCall}
continuous_actor/CastCastobservations*

DstT0*

SrcT0*
_output_shapes
:	?2
continuous_actor/Cast?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallcontinuous_actor/Cast:y:0continuous_actor_4888065continuous_actor_4888067continuous_actor_4888069continuous_actor_4888071continuous_actor_4888073continuous_actor_4888075continuous_actor_4888077continuous_actor_4888079continuous_actor_4888081continuous_actor_4888083continuous_actor_4888085continuous_actor_4888087continuous_actor_4888089continuous_actor_4888091continuous_actor_4888093continuous_actor_4888095continuous_actor_4888097continuous_actor_4888099continuous_actor_4888101*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_48880642*
(continuous_actor/StatefulPartitionedCallm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
Exp{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal/shape?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0Exp:y:0*
T0*
_output_shapes

:2
random_normal/mul?
random_normalAddV2random_normal/mul:z:01continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
random_normalP
TanhTanhrandom_normal:z:0*
T0*
_output_shapes

:2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yV
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes

:2
add?
mul/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L?2
mul/xS
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes

:2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yc
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes

:2	
truediv?
add_1/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾2	
add_1/x_
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
add_1[
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes

:2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:	?: : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:M I

_output_shapes
:	?
&
_user_specified_nameobservations
?
?
'__inference_mean_layer_call_fn_33438922

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_334376892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

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
މ
?
__inference_call_4886069

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_mean_layer_call_and_return_conditional_losses_33438932

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

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
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_33437350

inputs+
batch_normalization_33437288:	?+
batch_normalization_33437290:	?+
batch_normalization_33437292:	?+
batch_normalization_33437294:	?"
dense_33437309:
??
dense_33437311:	?-
batch_normalization_1_33437314:	?-
batch_normalization_1_33437316:	?-
batch_normalization_1_33437318:	?-
batch_normalization_1_33437320:	?$
dense_1_33437335:
??
dense_1_33437337:	?-
batch_normalization_2_33437340:	?-
batch_normalization_2_33437342:	?-
batch_normalization_2_33437344:	?-
batch_normalization_2_33437346:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_33437288batch_normalization_33437290batch_normalization_33437292batch_normalization_33437294*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368192-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_33437309dense_33437311*
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
C__inference_dense_layer_call_and_return_conditional_losses_334373082
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_33437314batch_normalization_1_33437316batch_normalization_1_33437318batch_normalization_1_33437320*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334369812/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_33437335dense_1_33437337*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_334373342!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_33437340batch_normalization_2_33437342batch_normalization_2_33437344batch_normalization_2_33437346*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334371432/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33437703

inputs"
sequential_33437646:	?"
sequential_33437648:	?"
sequential_33437650:	?"
sequential_33437652:	?'
sequential_33437654:
??"
sequential_33437656:	?"
sequential_33437658:	?"
sequential_33437660:	?"
sequential_33437662:	?"
sequential_33437664:	?'
sequential_33437666:
??"
sequential_33437668:	?"
sequential_33437670:	?"
sequential_33437672:	?"
sequential_33437674:	?"
sequential_33437676:	? 
mean_33437690:	?
mean_33437692:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_33437646sequential_33437648sequential_33437650sequential_33437652sequential_33437654sequential_33437656sequential_33437658sequential_33437660sequential_33437662sequential_33437664sequential_33437666sequential_33437668sequential_33437670sequential_33437672sequential_33437674sequential_33437676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334373502$
"sequential/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0mean_33437690mean_33437692*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_334376892
mean/StatefulPartitionedCall?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_value?
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?
H__inference_sequential_layer_call_and_return_conditional_losses_33438805

inputs?
0batch_normalization_cast_readvariableop_resource:	?A
2batch_normalization_cast_1_readvariableop_resource:	?A
2batch_normalization_cast_2_readvariableop_resource:	?A
2batch_normalization_cast_3_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?C
4batch_normalization_1_cast_2_readvariableop_resource:	?C
4batch_normalization_1_cast_3_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?A
2batch_normalization_2_cast_readvariableop_resource:	?C
4batch_normalization_2_cast_1_readvariableop_resource:	?C
4batch_normalization_2_cast_2_readvariableop_resource:	?C
4batch_normalization_2_cast_3_readvariableop_resource:	?
identity??'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?)batch_normalization/Cast_2/ReadVariableOp?)batch_normalization/Cast_3/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?+batch_normalization_2/Cast_2/ReadVariableOp?+batch_normalization_2/Cast_3/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp?
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/add_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
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
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/add_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/Relu?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp?
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOp?
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOp?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/Rsqrt?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/mul_1?
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/mul_2?
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_33438958

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368792
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_33439121

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
E__inference_dense_1_layer_call_and_return_conditional_losses_334373342
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
?
?
3__inference_continuous_actor_layer_call_fn_33438130
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_334377032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
B__inference_mean_layer_call_and_return_conditional_losses_33437689

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

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
?
?
8__inference_batch_normalization_1_layer_call_fn_33439045

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334369812
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ň
?
__inference_call_4888064

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueg
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:	?: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_33437385
batch_normalization_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334373502
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
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
(
_output_shapes
:??????????
3
_user_specified_namebatch_normalization_input
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439178

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_33437334

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
?
-__inference_sequential_layer_call_fn_33438739

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334374852
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
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
$__inference__traced_restore_33439360
file_prefix.
assignvariableop_log_std_dev:1
assignvariableop_1_mean_kernel:	?*
assignvariableop_2_mean_bias:;
,assignvariableop_3_batch_normalization_gamma:	?:
+assignvariableop_4_batch_normalization_beta:	?3
assignvariableop_5_dense_kernel:
??,
assignvariableop_6_dense_bias:	?=
.assignvariableop_7_batch_normalization_1_gamma:	?<
-assignvariableop_8_batch_normalization_1_beta:	?5
!assignvariableop_9_dense_1_kernel:
??/
 assignvariableop_10_dense_1_bias:	?>
/assignvariableop_11_batch_normalization_2_gamma:	?=
.assignvariableop_12_batch_normalization_2_beta:	?B
3assignvariableop_13_batch_normalization_moving_mean:	?F
7assignvariableop_14_batch_normalization_moving_variance:	?D
5assignvariableop_15_batch_normalization_1_moving_mean:	?H
9assignvariableop_16_batch_normalization_1_moving_variance:	?D
5assignvariableop_17_batch_normalization_2_moving_mean:	?H
9assignvariableop_18_batch_normalization_2_moving_variance:	?
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#_log_std/.ATTRIBUTES/VARIABLE_VALUEB%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_log_std_devIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_mean_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_mean_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batch_normalization_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_batch_normalization_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_1_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_1_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_batch_normalization_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_1_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_1_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_2_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_2_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19f
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_20?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?1
?	
!__inference__traced_save_33439293
file_prefix*
&savev2_log_std_dev_read_readvariableop*
&savev2_mean_kernel_read_readvariableop(
$savev2_mean_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#_log_std/.ATTRIBUTES/VARIABLE_VALUEB%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_log_std_dev_read_readvariableop&savev2_mean_kernel_read_readvariableop$savev2_mean_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::	?::?:?:
??:?:?:?:
??:?:?:?:?:?:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::%!

_output_shapes
:	?: 

_output_shapes
::!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: 
?
?
8__inference_batch_normalization_2_layer_call_fn_33439158

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334372032
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438544
input_1J
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinput_10sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
8__inference_batch_normalization_2_layer_call_fn_33439145

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334371432
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33436981

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_33438913

inputsJ
;batch_normalization_assignmovingavg_readvariableop_resource:	?L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	??
0batch_normalization_cast_readvariableop_resource:	?A
2batch_normalization_cast_1_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_2_cast_readvariableop_resource:	?C
4batch_normalization_2_cast_1_readvariableop_resource:	?
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	?2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/mul?
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg?
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/mul?
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/add_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMeandense/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/mul?
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg?
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/mul?
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/add_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/Relu?
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices?
"batch_normalization_2/moments/meanMeandense_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2$
"batch_normalization_2/moments/mean?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?2,
*batch_normalization_2/moments/StopGradient?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????21
/batch_normalization_2/moments/SquaredDifference?
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2(
&batch_normalization_2/moments/variance?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1?
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_2/AssignMovingAvg/decay?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_2/AssignMovingAvg/sub?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_2/AssignMovingAvg/mul?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg?
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_2/AssignMovingAvg_1/decay?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_2/AssignMovingAvg_1/sub?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_2/AssignMovingAvg_1/mul?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/Rsqrt?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/mul_1?
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/mul_2?
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_33438220

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_334378522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33437203

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_33438265
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_334378522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?%
?
H__inference_sequential_layer_call_and_return_conditional_losses_33437639
batch_normalization_input+
batch_normalization_33437601:	?+
batch_normalization_33437603:	?+
batch_normalization_33437605:	?+
batch_normalization_33437607:	?"
dense_33437610:
??
dense_33437612:	?-
batch_normalization_1_33437615:	?-
batch_normalization_1_33437617:	?-
batch_normalization_1_33437619:	?-
batch_normalization_1_33437621:	?$
dense_1_33437624:
??
dense_1_33437626:	?-
batch_normalization_2_33437629:	?-
batch_normalization_2_33437631:	?-
batch_normalization_2_33437633:	?-
batch_normalization_2_33437635:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_33437601batch_normalization_33437603batch_normalization_33437605batch_normalization_33437607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368792-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_33437610dense_33437612*
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
C__inference_dense_layer_call_and_return_conditional_losses_334373082
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_33437615batch_normalization_1_33437617batch_normalization_1_33437619batch_normalization_1_33437621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334370412/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_33437624dense_1_33437626*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_334373342!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_33437629batch_normalization_2_33437631batch_normalization_2_33437633batch_normalization_2_33437635*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334372032/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:c _
(
_output_shapes
:??????????
3
_user_specified_namebatch_normalization_input
ֈ
?
__inference_call_4888549

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueh
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	?2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:
??: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_33437485

inputs+
batch_normalization_33437447:	?+
batch_normalization_33437449:	?+
batch_normalization_33437451:	?+
batch_normalization_33437453:	?"
dense_33437456:
??
dense_33437458:	?-
batch_normalization_1_33437461:	?-
batch_normalization_1_33437463:	?-
batch_normalization_1_33437465:	?-
batch_normalization_1_33437467:	?$
dense_1_33437470:
??
dense_1_33437472:	?-
batch_normalization_2_33437475:	?-
batch_normalization_2_33437477:	?-
batch_normalization_2_33437479:	?-
batch_normalization_2_33437481:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_33437447batch_normalization_33437449batch_normalization_33437451batch_normalization_33437453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_334368792-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_33437456dense_33437458*
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
C__inference_dense_layer_call_and_return_conditional_losses_334373082
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_33437461batch_normalization_1_33437463batch_normalization_1_33437465batch_normalization_1_33437467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334370412/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_33437470dense_1_33437472*
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
E__inference_dense_1_layer_call_and_return_conditional_losses_334373342!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_33437475batch_normalization_2_33437477batch_normalization_2_33437479batch_normalization_2_33437481*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_334372032/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33439012

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_33439132

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
?
?
8__inference_batch_normalization_1_layer_call_fn_33439058

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_334370412
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
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439212

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438344

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_33439032

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
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33436819

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ֈ
?
__inference_call_4888201

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueh
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	?2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:
??: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
?
__inference_action_4888257
observations'
continuous_actor_4888202:	?'
continuous_actor_4888204:	?'
continuous_actor_4888206:	?'
continuous_actor_4888208:	?,
continuous_actor_4888210:
??'
continuous_actor_4888212:	?'
continuous_actor_4888214:	?'
continuous_actor_4888216:	?'
continuous_actor_4888218:	?'
continuous_actor_4888220:	?,
continuous_actor_4888222:
??'
continuous_actor_4888224:	?'
continuous_actor_4888226:	?'
continuous_actor_4888228:	?'
continuous_actor_4888230:	?'
continuous_actor_4888232:	?+
continuous_actor_4888234:	?&
continuous_actor_4888236:*
continuous_actor_4888238:
identity??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallobservationscontinuous_actor_4888202continuous_actor_4888204continuous_actor_4888206continuous_actor_4888208continuous_actor_4888210continuous_actor_4888212continuous_actor_4888214continuous_actor_4888216continuous_actor_4888218continuous_actor_4888220continuous_actor_4888222continuous_actor_4888224continuous_actor_4888226continuous_actor_4888228continuous_actor_4888230continuous_actor_4888232continuous_actor_4888234continuous_actor_4888236continuous_actor_4888238*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:	?:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_48882012*
(continuous_actor/StatefulPartitionedCallm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
Exp{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal/shape?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes
:	?*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0Exp:y:0*
T0*
_output_shapes
:	?2
random_normal/mul?
random_normalAddV2random_normal/mul:z:01continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
random_normalQ
TanhTanhrandom_normal:z:0*
T0*
_output_shapes
:	?2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes
:	?2
add?
mul/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L?2
mul/xT
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes
:	?2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yd
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes
:	?2	
truediv?
add_1/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾2	
add_1/x`
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes
:	?2
add_1\
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
:	?2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:
??: : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:N J
 
_output_shapes
:
??
&
_user_specified_nameobservations
?
?
3__inference_continuous_actor_layer_call_fn_33438175

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_334377032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_33438085
input_1
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_334367952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
-__inference_sequential_layer_call_fn_33437557
batch_normalization_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334374852
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
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
(
_output_shapes
:??????????
3
_user_specified_namebatch_normalization_input
?
?
-__inference_sequential_layer_call_fn_33438702

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334373502
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
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439078

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33437143

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33436879

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33437852

inputs"
sequential_33437806:	?"
sequential_33437808:	?"
sequential_33437810:	?"
sequential_33437812:	?'
sequential_33437814:
??"
sequential_33437816:	?"
sequential_33437818:	?"
sequential_33437820:	?"
sequential_33437822:	?"
sequential_33437824:	?'
sequential_33437826:
??"
sequential_33437828:	?"
sequential_33437830:	?"
sequential_33437832:	?"
sequential_33437834:	?"
sequential_33437836:	? 
mean_33437839:	?
mean_33437841:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_33437806sequential_33437808sequential_33437810sequential_33437812sequential_33437814sequential_33437816sequential_33437818sequential_33437820sequential_33437822sequential_33437824sequential_33437826sequential_33437828sequential_33437830sequential_33437832sequential_33437834sequential_33437836*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_334374852$
"sequential/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0mean_33437839mean_33437841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_334376892
mean/StatefulPartitionedCall?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_value?
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference__wrapped_model_33436795
input_1(
continuous_actor_33436753:	?(
continuous_actor_33436755:	?(
continuous_actor_33436757:	?(
continuous_actor_33436759:	?-
continuous_actor_33436761:
??(
continuous_actor_33436763:	?(
continuous_actor_33436765:	?(
continuous_actor_33436767:	?(
continuous_actor_33436769:	?(
continuous_actor_33436771:	?-
continuous_actor_33436773:
??(
continuous_actor_33436775:	?(
continuous_actor_33436777:	?(
continuous_actor_33436779:	?(
continuous_actor_33436781:	?(
continuous_actor_33436783:	?,
continuous_actor_33436785:	?'
continuous_actor_33436787:+
continuous_actor_33436789:
identity

identity_1??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallinput_1continuous_actor_33436753continuous_actor_33436755continuous_actor_33436757continuous_actor_33436759continuous_actor_33436761continuous_actor_33436763continuous_actor_33436765continuous_actor_33436767continuous_actor_33436769continuous_actor_33436771continuous_actor_33436773continuous_actor_33436775continuous_actor_33436777continuous_actor_33436779continuous_actor_33436781continuous_actor_33436783continuous_actor_33436785continuous_actor_33436787continuous_actor_33436789*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????:*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_48860692*
(continuous_actor/StatefulPartitionedCall?
IdentityIdentity1continuous_actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1continuous_actor/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1y
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438465

inputsU
Fsequential_batch_normalization_assignmovingavg_readvariableop_resource:	?W
Hsequential_batch_normalization_assignmovingavg_1_readvariableop_resource:	?J
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_1_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_2_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?.sequential/batch_normalization/AssignMovingAvg?=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?0sequential/batch_normalization/AssignMovingAvg_1??sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?0sequential/batch_normalization_1/AssignMovingAvg??sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_1/AssignMovingAvg_1?Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?0sequential/batch_normalization_2/AssignMovingAvg??sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_2/AssignMovingAvg_1?Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
=sequential/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential/batch_normalization/moments/mean/reduction_indices?
+sequential/batch_normalization/moments/meanMeaninputsFsequential/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2-
+sequential/batch_normalization/moments/mean?
3sequential/batch_normalization/moments/StopGradientStopGradient4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	?25
3sequential/batch_normalization/moments/StopGradient?
8sequential/batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs<sequential/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2:
8sequential/batch_normalization/moments/SquaredDifference?
Asequential/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/batch_normalization/moments/variance/reduction_indices?
/sequential/batch_normalization/moments/varianceMean<sequential/batch_normalization/moments/SquaredDifference:z:0Jsequential/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(21
/sequential/batch_normalization/moments/variance?
.sequential/batch_normalization/moments/SqueezeSqueeze4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 20
.sequential/batch_normalization/moments/Squeeze?
0sequential/batch_normalization/moments/Squeeze_1Squeeze8sequential/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization/moments/Squeeze_1?
4sequential/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential/batch_normalization/AssignMovingAvg/decay?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?
2sequential/batch_normalization/AssignMovingAvg/subSubEsequential/batch_normalization/AssignMovingAvg/ReadVariableOp:value:07sequential/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:?24
2sequential/batch_normalization/AssignMovingAvg/sub?
2sequential/batch_normalization/AssignMovingAvg/mulMul6sequential/batch_normalization/AssignMovingAvg/sub:z:0=sequential/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?24
2sequential/batch_normalization/AssignMovingAvg/mul?
.sequential/batch_normalization/AssignMovingAvgAssignSubVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource6sequential/batch_normalization/AssignMovingAvg/mul:z:0>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype020
.sequential/batch_normalization/AssignMovingAvg?
6sequential/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization/AssignMovingAvg_1/decay?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?
4sequential/batch_normalization/AssignMovingAvg_1/subSubGsequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:09sequential/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization/AssignMovingAvg_1/sub?
4sequential/batch_normalization/AssignMovingAvg_1/mulMul8sequential/batch_normalization/AssignMovingAvg_1/sub:z:0?sequential/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization/AssignMovingAvg_1/mul?
0sequential/batch_normalization/AssignMovingAvg_1AssignSubVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource8sequential/batch_normalization/AssignMovingAvg_1/mul:z:0@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization/AssignMovingAvg_1?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV29sequential/batch_normalization/moments/Squeeze_1:output:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul7sequential/batch_normalization/moments/Squeeze:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub:sequential/batch_normalization/Cast/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
?sequential/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_1/moments/mean/reduction_indices?
-sequential/batch_normalization_1/moments/meanMean#sequential/dense/Relu:activations:0Hsequential/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_1/moments/mean?
5sequential/batch_normalization_1/moments/StopGradientStopGradient6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_1/moments/StopGradient?
:sequential/batch_normalization_1/moments/SquaredDifferenceSquaredDifference#sequential/dense/Relu:activations:0>sequential/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_1/moments/SquaredDifference?
Csequential/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_1/moments/variance/reduction_indices?
1sequential/batch_normalization_1/moments/varianceMean>sequential/batch_normalization_1/moments/SquaredDifference:z:0Lsequential/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_1/moments/variance?
0sequential/batch_normalization_1/moments/SqueezeSqueeze6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_1/moments/Squeeze?
2sequential/batch_normalization_1/moments/Squeeze_1Squeeze:sequential/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_1/moments/Squeeze_1?
6sequential/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_1/AssignMovingAvg/decay?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_1/AssignMovingAvg/subSubGsequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/sub?
4sequential/batch_normalization_1/AssignMovingAvg/mulMul8sequential/batch_normalization_1/AssignMovingAvg/sub:z:0?sequential/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/mul?
0sequential/batch_normalization_1/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource8sequential/batch_normalization_1/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_1/AssignMovingAvg?
8sequential/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_1/AssignMovingAvg_1/decay?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_1/AssignMovingAvg_1/subSubIsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/sub?
6sequential/batch_normalization_1/AssignMovingAvg_1/mulMul:sequential/batch_normalization_1/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/mul?
2sequential/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_1/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_1/AssignMovingAvg_1?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2;sequential/batch_normalization_1/moments/Squeeze_1:output:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul9sequential/batch_normalization_1/moments/Squeeze:output:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub<sequential/batch_normalization_1/Cast/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
?sequential/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_2/moments/mean/reduction_indices?
-sequential/batch_normalization_2/moments/meanMean%sequential/dense_1/Relu:activations:0Hsequential/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_2/moments/mean?
5sequential/batch_normalization_2/moments/StopGradientStopGradient6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_2/moments/StopGradient?
:sequential/batch_normalization_2/moments/SquaredDifferenceSquaredDifference%sequential/dense_1/Relu:activations:0>sequential/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_2/moments/SquaredDifference?
Csequential/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_2/moments/variance/reduction_indices?
1sequential/batch_normalization_2/moments/varianceMean>sequential/batch_normalization_2/moments/SquaredDifference:z:0Lsequential/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_2/moments/variance?
0sequential/batch_normalization_2/moments/SqueezeSqueeze6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_2/moments/Squeeze?
2sequential/batch_normalization_2/moments/Squeeze_1Squeeze:sequential/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_2/moments/Squeeze_1?
6sequential/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_2/AssignMovingAvg/decay?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_2/AssignMovingAvg/subSubGsequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/sub?
4sequential/batch_normalization_2/AssignMovingAvg/mulMul8sequential/batch_normalization_2/AssignMovingAvg/sub:z:0?sequential/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/mul?
0sequential/batch_normalization_2/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource8sequential/batch_normalization_2/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_2/AssignMovingAvg?
8sequential/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_2/AssignMovingAvg_1/decay?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_2/AssignMovingAvg_1/subSubIsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/sub?
6sequential/batch_normalization_2/AssignMovingAvg_1/mulMul:sequential/batch_normalization_2/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/mul?
2sequential/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_2/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_2/AssignMovingAvg_1?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2;sequential/batch_normalization_2/moments/Squeeze_1:output:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul9sequential/batch_normalization_2/moments/Squeeze:output:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub<sequential/batch_normalization_2/Cast/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?

NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp/^sequential/batch_normalization/AssignMovingAvg>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp1^sequential/batch_normalization/AssignMovingAvg_1@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp1^sequential/batch_normalization_1/AssignMovingAvg@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_1/AssignMovingAvg_1B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp1^sequential/batch_normalization_2/AssignMovingAvg@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_2/AssignMovingAvg_1B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2`
.sequential/batch_normalization/AssignMovingAvg.sequential/batch_normalization/AssignMovingAvg2~
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp2d
0sequential/batch_normalization/AssignMovingAvg_10sequential/batch_normalization/AssignMovingAvg_12?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_1/AssignMovingAvg0sequential/batch_normalization_1/AssignMovingAvg2?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_1/AssignMovingAvg_12sequential/batch_normalization_1/AssignMovingAvg_12?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_2/AssignMovingAvg0sequential/batch_normalization_2/AssignMovingAvg2?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_2/AssignMovingAvg_12sequential/batch_normalization_2/AssignMovingAvg_12?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ň
?
__inference_call_4888470

inputsJ
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?L
=sequential_batch_normalization_cast_2_readvariableop_resource:	?L
=sequential_batch_normalization_cast_3_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueg
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:	?: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438665
input_1U
Fsequential_batch_normalization_assignmovingavg_readvariableop_resource:	?W
Hsequential_batch_normalization_assignmovingavg_1_readvariableop_resource:	?J
;sequential_batch_normalization_cast_readvariableop_resource:	?L
=sequential_batch_normalization_cast_1_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_1_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_2_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:7
%clip_by_value_readvariableop_resource:
identity

identity_1??clip_by_value/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?.sequential/batch_normalization/AssignMovingAvg?=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?0sequential/batch_normalization/AssignMovingAvg_1??sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?0sequential/batch_normalization_1/AssignMovingAvg??sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_1/AssignMovingAvg_1?Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?0sequential/batch_normalization_2/AssignMovingAvg??sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_2/AssignMovingAvg_1?Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
=sequential/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential/batch_normalization/moments/mean/reduction_indices?
+sequential/batch_normalization/moments/meanMeaninput_1Fsequential/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2-
+sequential/batch_normalization/moments/mean?
3sequential/batch_normalization/moments/StopGradientStopGradient4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	?25
3sequential/batch_normalization/moments/StopGradient?
8sequential/batch_normalization/moments/SquaredDifferenceSquaredDifferenceinput_1<sequential/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2:
8sequential/batch_normalization/moments/SquaredDifference?
Asequential/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/batch_normalization/moments/variance/reduction_indices?
/sequential/batch_normalization/moments/varianceMean<sequential/batch_normalization/moments/SquaredDifference:z:0Jsequential/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(21
/sequential/batch_normalization/moments/variance?
.sequential/batch_normalization/moments/SqueezeSqueeze4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 20
.sequential/batch_normalization/moments/Squeeze?
0sequential/batch_normalization/moments/Squeeze_1Squeeze8sequential/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization/moments/Squeeze_1?
4sequential/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential/batch_normalization/AssignMovingAvg/decay?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?
2sequential/batch_normalization/AssignMovingAvg/subSubEsequential/batch_normalization/AssignMovingAvg/ReadVariableOp:value:07sequential/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:?24
2sequential/batch_normalization/AssignMovingAvg/sub?
2sequential/batch_normalization/AssignMovingAvg/mulMul6sequential/batch_normalization/AssignMovingAvg/sub:z:0=sequential/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?24
2sequential/batch_normalization/AssignMovingAvg/mul?
.sequential/batch_normalization/AssignMovingAvgAssignSubVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource6sequential/batch_normalization/AssignMovingAvg/mul:z:0>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype020
.sequential/batch_normalization/AssignMovingAvg?
6sequential/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization/AssignMovingAvg_1/decay?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?
4sequential/batch_normalization/AssignMovingAvg_1/subSubGsequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:09sequential/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization/AssignMovingAvg_1/sub?
4sequential/batch_normalization/AssignMovingAvg_1/mulMul8sequential/batch_normalization/AssignMovingAvg_1/sub:z:0?sequential/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization/AssignMovingAvg_1/mul?
0sequential/batch_normalization/AssignMovingAvg_1AssignSubVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource8sequential/batch_normalization/AssignMovingAvg_1/mul:z:0@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization/AssignMovingAvg_1?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV29sequential/batch_normalization/moments/Squeeze_1:output:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinput_10sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul7sequential/batch_normalization/moments/Squeeze:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub:sequential/batch_normalization/Cast/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
?sequential/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_1/moments/mean/reduction_indices?
-sequential/batch_normalization_1/moments/meanMean#sequential/dense/Relu:activations:0Hsequential/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_1/moments/mean?
5sequential/batch_normalization_1/moments/StopGradientStopGradient6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_1/moments/StopGradient?
:sequential/batch_normalization_1/moments/SquaredDifferenceSquaredDifference#sequential/dense/Relu:activations:0>sequential/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_1/moments/SquaredDifference?
Csequential/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_1/moments/variance/reduction_indices?
1sequential/batch_normalization_1/moments/varianceMean>sequential/batch_normalization_1/moments/SquaredDifference:z:0Lsequential/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_1/moments/variance?
0sequential/batch_normalization_1/moments/SqueezeSqueeze6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_1/moments/Squeeze?
2sequential/batch_normalization_1/moments/Squeeze_1Squeeze:sequential/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_1/moments/Squeeze_1?
6sequential/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_1/AssignMovingAvg/decay?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_1/AssignMovingAvg/subSubGsequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/sub?
4sequential/batch_normalization_1/AssignMovingAvg/mulMul8sequential/batch_normalization_1/AssignMovingAvg/sub:z:0?sequential/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/mul?
0sequential/batch_normalization_1/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource8sequential/batch_normalization_1/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_1/AssignMovingAvg?
8sequential/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_1/AssignMovingAvg_1/decay?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_1/AssignMovingAvg_1/subSubIsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/sub?
6sequential/batch_normalization_1/AssignMovingAvg_1/mulMul:sequential/batch_normalization_1/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/mul?
2sequential/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_1/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_1/AssignMovingAvg_1?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2;sequential/batch_normalization_1/moments/Squeeze_1:output:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul9sequential/batch_normalization_1/moments/Squeeze:output:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub<sequential/batch_normalization_1/Cast/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
?sequential/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_2/moments/mean/reduction_indices?
-sequential/batch_normalization_2/moments/meanMean%sequential/dense_1/Relu:activations:0Hsequential/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_2/moments/mean?
5sequential/batch_normalization_2/moments/StopGradientStopGradient6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_2/moments/StopGradient?
:sequential/batch_normalization_2/moments/SquaredDifferenceSquaredDifference%sequential/dense_1/Relu:activations:0>sequential/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_2/moments/SquaredDifference?
Csequential/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_2/moments/variance/reduction_indices?
1sequential/batch_normalization_2/moments/varianceMean>sequential/batch_normalization_2/moments/SquaredDifference:z:0Lsequential/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_2/moments/variance?
0sequential/batch_normalization_2/moments/SqueezeSqueeze6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_2/moments/Squeeze?
2sequential/batch_normalization_2/moments/Squeeze_1Squeeze:sequential/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_2/moments/Squeeze_1?
6sequential/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_2/AssignMovingAvg/decay?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_2/AssignMovingAvg/subSubGsequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/sub?
4sequential/batch_normalization_2/AssignMovingAvg/mulMul8sequential/batch_normalization_2/AssignMovingAvg/sub:z:0?sequential/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/mul?
0sequential/batch_normalization_2/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource8sequential/batch_normalization_2/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_2/AssignMovingAvg?
8sequential/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_2/AssignMovingAvg_1/decay?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_2/AssignMovingAvg_1/subSubIsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/sub?
6sequential/batch_normalization_2/AssignMovingAvg_1/mulMul:sequential/batch_normalization_2/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/mul?
2sequential/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_2/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_2/AssignMovingAvg_1?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2;sequential/batch_normalization_2/moments/Squeeze_1:output:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul9sequential/batch_normalization_2/moments/Squeeze:output:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub<sequential/batch_normalization_2/Cast/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*
_output_shapes

:*
dtype02
clip_by_value/ReadVariableOpw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?

NoOpNoOp^clip_by_value/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp/^sequential/batch_normalization/AssignMovingAvg>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp1^sequential/batch_normalization/AssignMovingAvg_1@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp1^sequential/batch_normalization_1/AssignMovingAvg@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_1/AssignMovingAvg_1B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp1^sequential/batch_normalization_2/AssignMovingAvg@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_2/AssignMovingAvg_1B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????: : : : : : : : : : : : : : : : : : : 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2`
.sequential/batch_normalization/AssignMovingAvg.sequential/batch_normalization/AssignMovingAvg2~
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp2d
0sequential/batch_normalization/AssignMovingAvg_10sequential/batch_normalization/AssignMovingAvg_12?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_1/AssignMovingAvg0sequential/batch_normalization_1/AssignMovingAvg2?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_1/AssignMovingAvg_12sequential/batch_normalization_1/AssignMovingAvg_12?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_2/AssignMovingAvg0sequential/batch_normalization_2/AssignMovingAvg2?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_2/AssignMovingAvg_12sequential/batch_normalization_2/AssignMovingAvg_12?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
__inference_action_4888312
observations'
continuous_actor_4888261:	?'
continuous_actor_4888263:	?'
continuous_actor_4888265:	?'
continuous_actor_4888267:	?,
continuous_actor_4888269:
??'
continuous_actor_4888271:	?'
continuous_actor_4888273:	?'
continuous_actor_4888275:	?'
continuous_actor_4888277:	?'
continuous_actor_4888279:	?,
continuous_actor_4888281:
??'
continuous_actor_4888283:	?'
continuous_actor_4888285:	?'
continuous_actor_4888287:	?'
continuous_actor_4888289:	?'
continuous_actor_4888291:	?+
continuous_actor_4888293:	?&
continuous_actor_4888295:*
continuous_actor_4888297:
identity??(continuous_actor/StatefulPartitionedCall}
continuous_actor/CastCastobservations*

DstT0*

SrcT0*
_output_shapes
:	?2
continuous_actor/Cast?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallcontinuous_actor/Cast:y:0continuous_actor_4888261continuous_actor_4888263continuous_actor_4888265continuous_actor_4888267continuous_actor_4888269continuous_actor_4888271continuous_actor_4888273continuous_actor_4888275continuous_actor_4888277continuous_actor_4888279continuous_actor_4888281continuous_actor_4888283continuous_actor_4888285continuous_actor_4888287continuous_actor_4888289continuous_actor_4888291continuous_actor_4888293continuous_actor_4888295continuous_actor_4888297*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_48880642*
(continuous_actor/StatefulPartitionedCallm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
Expp
TanhTanh1continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yV
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes

:2
add?
mul/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L???L?2
mul/xS
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes

:2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yc
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes

:2	
truediv?
add_1/xConst*
_output_shapes
:*
dtype0*Y
valuePBN"D??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾??̾2	
add_1/x_
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
add_1[
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes

:2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:	?: : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:M I

_output_shapes
:	?
&
_user_specified_nameobservations
?
?
(__inference_dense_layer_call_fn_33439021

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
GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_334373082
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????<
output_10
StatefulPartitionedCall:0?????????3
output_2'
StatefulPartitionedCall:1tensorflow/serving/predict:??
?
	_body
_mu
_log_std
trainable_variables
	variables
regularization_losses
	keras_api

signatures
g__call__
h_default_save_signature
*i&call_and_return_all_conditional_losses

jaction
kcall
llogprob
msample_logprob"
_tf_keras_model
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
trainable_variables
	variables
regularization_losses
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
:2log_std_dev
~
0
1
2
3
4
5
6
7
 8
!9
10
11
12"
trackable_list_wrapper
?
0
1
"2
#3
4
5
6
7
$8
%9
10
11
 12
!13
&14
'15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(metrics
trainable_variables
	variables
regularization_losses
)non_trainable_variables

*layers
+layer_metrics
,layer_regularization_losses
g__call__
h_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
?
-axis
	gamma
beta
"moving_mean
#moving_variance
.trainable_variables
/	variables
0regularization_losses
1	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6axis
	gamma
beta
$moving_mean
%moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?axis
	 gamma
!beta
&moving_mean
'moving_variance
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
 8
!9"
trackable_list_wrapper
?
0
1
"2
#3
4
5
6
7
$8
%9
10
11
 12
!13
&14
'15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dmetrics
trainable_variables
	variables
regularization_losses
Enon_trainable_variables

Flayers
Glayer_metrics
Hlayer_regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	?2mean/kernel
:2	mean/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses

Klayers
Llayer_metrics
Mmetrics
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
 :
??2dense/kernel
:?2
dense/bias
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
": 
??2dense_1/kernel
:?2dense_1/bias
*:(?2batch_normalization_2/gamma
):'?2batch_normalization_2/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
2:0? (2!batch_normalization_2/moving_mean
6:4? (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
"2
#3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.trainable_variables
/	variables
0regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses

Players
Qlayer_metrics
Rmetrics
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2trainable_variables
3	variables
4regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
Vlayer_metrics
Wmetrics
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7trainable_variables
8	variables
9regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
[layer_metrics
\metrics
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;trainable_variables
<	variables
=regularization_losses
]non_trainable_variables
^layer_regularization_losses

_layers
`layer_metrics
ametrics
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
&2
'3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@trainable_variables
A	variables
Bregularization_losses
bnon_trainable_variables
clayer_regularization_losses

dlayers
elayer_metrics
fmetrics
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
C
	0

1
2
3
4"
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
.
"0
#1"
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
.
$0
%1"
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
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
3__inference_continuous_actor_layer_call_fn_33438130
3__inference_continuous_actor_layer_call_fn_33438175
3__inference_continuous_actor_layer_call_fn_33438220
3__inference_continuous_actor_layer_call_fn_33438265?
???
FullArgSpec)
args!?
jself
jinputs

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
#__inference__wrapped_model_33436795input_1"?
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
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438344
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438465
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438544
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438665?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
__inference_action_4888120
__inference_action_4888257
__inference_action_4888312?
???
FullArgSpec4
args,?)
jself
jobservations
jdeterministic
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
?2?
__inference_call_4888391
__inference_call_4888470
__inference_call_4888549?
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
__inference_logprob_4888610?
???
FullArgSpec,
args$?!
jself
jobservations
jvalue
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
?2??
???
FullArgSpec0
args(?%
jself
jobservations
j	n_samples
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
?2?
-__inference_sequential_layer_call_fn_33437385
-__inference_sequential_layer_call_fn_33438702
-__inference_sequential_layer_call_fn_33438739
-__inference_sequential_layer_call_fn_33437557?
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
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_33438805
H__inference_sequential_layer_call_and_return_conditional_losses_33438913
H__inference_sequential_layer_call_and_return_conditional_losses_33437598
H__inference_sequential_layer_call_and_return_conditional_losses_33437639?
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
'__inference_mean_layer_call_fn_33438922?
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
B__inference_mean_layer_call_and_return_conditional_losses_33438932?
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
&__inference_signature_wrapper_33438085input_1"?
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
?2?
6__inference_batch_normalization_layer_call_fn_33438945
6__inference_batch_normalization_layer_call_fn_33438958?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33438978
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33439012?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_33439021?
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
C__inference_dense_layer_call_and_return_conditional_losses_33439032?
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
?2?
8__inference_batch_normalization_1_layer_call_fn_33439045
8__inference_batch_normalization_1_layer_call_fn_33439058?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439078
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439112?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_33439121?
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
E__inference_dense_1_layer_call_and_return_conditional_losses_33439132?
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
?2?
8__inference_batch_normalization_2_layer_call_fn_33439145
8__inference_batch_normalization_2_layer_call_fn_33439158?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439178
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439212?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
#__inference__wrapped_model_33436795?"#$%&'! 1?.
'?$
"?
input_1??????????
? "Z?W
.
output_1"?
output_1?????????
%
output_2?
output_2w
__inference_action_4888120Y"#$%&'! 1?.
'?$
?
observations	?
p 
? "?y
__inference_action_4888257["#$%&'! 2?/
(?%
?
observations
??
p 
? "?	?w
__inference_action_4888312Y"#$%&'! 1?.
'?$
?
observations	?
p
? "??
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439078d$%4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_33439112d$%4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_1_layer_call_fn_33439045W$%4?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_1_layer_call_fn_33439058W$%4?1
*?'
!?
inputs??????????
p
? "????????????
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439178d&'! 4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_33439212d&'! 4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_2_layer_call_fn_33439145W&'! 4?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_2_layer_call_fn_33439158W&'! 4?1
*?'
!?
inputs??????????
p
? "????????????
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33438978d"#4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_33439012d"#4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
6__inference_batch_normalization_layer_call_fn_33438945W"#4?1
*?'
!?
inputs??????????
p 
? "????????????
6__inference_batch_normalization_layer_call_fn_33438958W"#4?1
*?'
!?
inputs??????????
p
? "????????????
__inference_call_4888391}"#$%&'! 0?-
&?#
!?
inputs??????????
? "4?1
?
0?????????
?
1?
__inference_call_4888470k"#$%&'! '?$
?
?
inputs	?
? "+?(
?
0
?
1?
__inference_call_4888549m"#$%&'! (?%
?
?
inputs
??
? ",?)
?
0	?
?
1?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438344?"#$%&'! 4?1
*?'
!?
inputs??????????
p 
? "B??
8?5
?
0/0?????????
?
0/1
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438465?"#$%&'! 4?1
*?'
!?
inputs??????????
p
? "B??
8?5
?
0/0?????????
?
0/1
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438544?"#$%&'! 5?2
+?(
"?
input_1??????????
p 
? "B??
8?5
?
0/0?????????
?
0/1
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_33438665?"#$%&'! 5?2
+?(
"?
input_1??????????
p
? "B??
8?5
?
0/0?????????
?
0/1
? ?
3__inference_continuous_actor_layer_call_fn_33438130?"#$%&'! 5?2
+?(
"?
input_1??????????
p 
? "4?1
?
0?????????
?
1?
3__inference_continuous_actor_layer_call_fn_33438175?"#$%&'! 4?1
*?'
!?
inputs??????????
p 
? "4?1
?
0?????????
?
1?
3__inference_continuous_actor_layer_call_fn_33438220?"#$%&'! 4?1
*?'
!?
inputs??????????
p
? "4?1
?
0?????????
?
1?
3__inference_continuous_actor_layer_call_fn_33438265?"#$%&'! 5?2
+?(
"?
input_1??????????
p
? "4?1
?
0?????????
?
1?
E__inference_dense_1_layer_call_and_return_conditional_losses_33439132^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_1_layer_call_fn_33439121Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_33439032^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_layer_call_fn_33439021Q0?-
&?#
!?
inputs??????????
? "????????????
__inference_logprob_4888610t"#$%&'! K?H
A?>
?
observations
??
?
value
?
? "?	
??
B__inference_mean_layer_call_and_return_conditional_losses_33438932]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_mean_layer_call_fn_33438922P0?-
&?#
!?
inputs??????????
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_33437598?"#$%&'! K?H
A?>
4?1
batch_normalization_input??????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_33437639?"#$%&'! K?H
A?>
4?1
batch_normalization_input??????????
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_33438805t"#$%&'! 8?5
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
H__inference_sequential_layer_call_and_return_conditional_losses_33438913t"#$%&'! 8?5
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
-__inference_sequential_layer_call_fn_33437385z"#$%&'! K?H
A?>
4?1
batch_normalization_input??????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_33437557z"#$%&'! K?H
A?>
4?1
batch_normalization_input??????????
p

 
? "????????????
-__inference_sequential_layer_call_fn_33438702g"#$%&'! 8?5
.?+
!?
inputs??????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_33438739g"#$%&'! 8?5
.?+
!?
inputs??????????
p

 
? "????????????
&__inference_signature_wrapper_33438085?"#$%&'! <?9
? 
2?/
-
input_1"?
input_1??????????"Z?W
.
output_1"?
output_1?????????
%
output_2?
output_2