¤·
«ü
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
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
list(type)(0
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
¾
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
executor_typestring 
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

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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8ð
s
mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namemean/kernel
l
mean/kernel/Read/ReadVariableOpReadVariableOpmean/kernel*
_output_shapes
:	*
dtype0
j
	mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	mean/bias
c
mean/bias/Read/ReadVariableOpReadVariableOp	mean/bias*
_output_shapes
:*
dtype0

log_std_dev/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namelog_std_dev/kernel
z
&log_std_dev/kernel/Read/ReadVariableOpReadVariableOplog_std_dev/kernel*
_output_shapes
:	*
dtype0
x
log_std_dev/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namelog_std_dev/bias
q
$log_std_dev/bias/Read/ReadVariableOpReadVariableOplog_std_dev/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0

NoOpNoOp
¢-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ý,
valueÓ,BÐ, BÉ,

	_body
_mu
_log_std
trainable_variables
	variables
regularization_losses
	keras_api

signatures

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
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9
10
11
12
13

0
1
(2
)3
 4
!5
"6
#7
*8
+9
$10
%11
&12
'13
,14
-15
16
17
18
19
 
­
trainable_variables
.layer_regularization_losses
/layer_metrics
0non_trainable_variables
	variables
regularization_losses
1metrics

2layers
 

3axis
	gamma
beta
(moving_mean
)moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

 kernel
!bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api

<axis
	"gamma
#beta
*moving_mean
+moving_variance
=trainable_variables
>	variables
?regularization_losses
@	keras_api
h

$kernel
%bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api

Eaxis
	&gamma
'beta
,moving_mean
-moving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
F
0
1
 2
!3
"4
#5
$6
%7
&8
'9
v
0
1
(2
)3
 4
!5
"6
#7
*8
+9
$10
%11
&12
'13
,14
-15
 
­
trainable_variables
Jlayer_regularization_losses
Klayer_metrics
Lnon_trainable_variables
	variables
regularization_losses
Mmetrics

Nlayers
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
­
trainable_variables
Olayer_regularization_losses
Player_metrics
Qnon_trainable_variables
	variables
regularization_losses
Rmetrics

Slayers
RP
VARIABLE_VALUElog_std_dev/kernel*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUElog_std_dev/bias(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
Vnon_trainable_variables
	variables
regularization_losses
Wmetrics

Xlayers
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
 
*
(0
)1
*2
+3
,4
-5
 

0
1
2
 

0
1

0
1
(2
)3
 
­
4trainable_variables
Ylayer_regularization_losses
Zlayer_metrics
[non_trainable_variables
5	variables
6regularization_losses
\metrics

]layers

 0
!1

 0
!1
 
­
8trainable_variables
^layer_regularization_losses
_layer_metrics
`non_trainable_variables
9	variables
:regularization_losses
ametrics

blayers
 

"0
#1

"0
#1
*2
+3
 
­
=trainable_variables
clayer_regularization_losses
dlayer_metrics
enon_trainable_variables
>	variables
?regularization_losses
fmetrics

glayers

$0
%1

$0
%1
 
­
Atrainable_variables
hlayer_regularization_losses
ilayer_metrics
jnon_trainable_variables
B	variables
Cregularization_losses
kmetrics

llayers
 

&0
'1

&0
'1
,2
-3
 
­
Ftrainable_variables
mlayer_regularization_losses
nlayer_metrics
onon_trainable_variables
G	variables
Hregularization_losses
pmetrics

qlayers
 
 
*
(0
)1
*2
+3
,4
-5
 
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
 
 
 
 
 

(0
)1
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
*0
+1
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
,0
-1
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
÷
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betamean/kernel	mean/biaslog_std_dev/kernellog_std_dev/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26718307
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/kernel/Read/ReadVariableOpmean/bias/Read/ReadVariableOp&log_std_dev/kernel/Read/ReadVariableOp$log_std_dev/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpConst*!
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_26719742
£
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemean/kernel	mean/biaslog_std_dev/kernellog_std_dev/biasbatch_normalization/gammabatch_normalization/betadense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/betadense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance* 
Tin
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_26719812¡
ï

(__inference_actor_layer_call_fn_26717554
batch_normalization_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	
identity¢StatefulPartitionedCall¼
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267174822
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namebatch_normalization_input

¶
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719497

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_2_layer_call_fn_26719657

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267172002
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê*
ð
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26717200

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À$
ã
C__inference_actor_layer_call_and_return_conditional_losses_26717347

inputs*
batch_normalization_26717285:*
batch_normalization_26717287:*
batch_normalization_26717289:*
batch_normalization_26717291:!
dense_26717306:	
dense_26717308:	-
batch_normalization_1_26717311:	-
batch_normalization_1_26717313:	-
batch_normalization_1_26717315:	-
batch_normalization_1_26717317:	$
dense_1_26717332:

dense_1_26717334:	-
batch_normalization_2_26717337:	-
batch_normalization_2_26717339:	-
batch_normalization_2_26717341:	-
batch_normalization_2_26717343:	
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_26717285batch_normalization_26717287batch_normalization_26717289batch_normalization_26717291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168162-
+batch_normalization/StatefulPartitionedCallº
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26717306dense_26717308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_267173052
dense/StatefulPartitionedCallÀ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_26717311batch_normalization_1_26717313batch_normalization_1_26717315batch_normalization_1_26717317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267169782/
-batch_normalization_1/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_26717332dense_1_26717334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_267173312!
dense_1/StatefulPartitionedCallÂ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_26717337batch_normalization_2_26717339batch_normalization_2_26717341batch_normalization_2_26717343*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267171402/
-batch_normalization_2/StatefulPartitionedCall
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

(__inference_actor_layer_call_fn_26719302

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	
identity¢StatefulPartitionedCall¯
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
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267173472
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ô
B__inference_mean_layer_call_and_return_conditional_losses_26717686

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
µ
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718895
input_1O
Aactor_batch_normalization_assignmovingavg_readvariableop_resource:Q
Cactor_batch_normalization_assignmovingavg_1_readvariableop_resource:M
?actor_batch_normalization_batchnorm_mul_readvariableop_resource:I
;actor_batch_normalization_batchnorm_readvariableop_resource:=
*actor_dense_matmul_readvariableop_resource:	:
+actor_dense_biasadd_readvariableop_resource:	R
Cactor_batch_normalization_1_assignmovingavg_readvariableop_resource:	T
Eactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	P
Aactor_batch_normalization_1_batchnorm_mul_readvariableop_resource:	L
=actor_batch_normalization_1_batchnorm_readvariableop_resource:	@
,actor_dense_1_matmul_readvariableop_resource:
<
-actor_dense_1_biasadd_readvariableop_resource:	R
Cactor_batch_normalization_2_assignmovingavg_readvariableop_resource:	T
Eactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	P
Aactor_batch_normalization_2_batchnorm_mul_readvariableop_resource:	L
=actor_batch_normalization_2_batchnorm_readvariableop_resource:	6
#mean_matmul_readvariableop_resource:	2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢)actor/batch_normalization/AssignMovingAvg¢8actor/batch_normalization/AssignMovingAvg/ReadVariableOp¢+actor/batch_normalization/AssignMovingAvg_1¢:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp¢2actor/batch_normalization/batchnorm/ReadVariableOp¢6actor/batch_normalization/batchnorm/mul/ReadVariableOp¢+actor/batch_normalization_1/AssignMovingAvg¢:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp¢-actor/batch_normalization_1/AssignMovingAvg_1¢<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢4actor/batch_normalization_1/batchnorm/ReadVariableOp¢8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp¢+actor/batch_normalization_2/AssignMovingAvg¢:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp¢-actor/batch_normalization_2/AssignMovingAvg_1¢<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢4actor/batch_normalization_2/batchnorm/ReadVariableOp¢8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"actor/dense/BiasAdd/ReadVariableOp¢!actor/dense/MatMul/ReadVariableOp¢$actor/dense_1/BiasAdd/ReadVariableOp¢#actor/dense_1/MatMul/ReadVariableOp¢"log_std_dev/BiasAdd/ReadVariableOp¢!log_std_dev/MatMul/ReadVariableOp¢mean/BiasAdd/ReadVariableOp¢mean/MatMul/ReadVariableOp¾
8actor/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8actor/batch_normalization/moments/mean/reduction_indicesÞ
&actor/batch_normalization/moments/meanMeaninput_1Aactor/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&actor/batch_normalization/moments/meanÊ
.actor/batch_normalization/moments/StopGradientStopGradient/actor/batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:20
.actor/batch_normalization/moments/StopGradientó
3actor/batch_normalization/moments/SquaredDifferenceSquaredDifferenceinput_17actor/batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3actor/batch_normalization/moments/SquaredDifferenceÆ
<actor/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2>
<actor/batch_normalization/moments/variance/reduction_indices
*actor/batch_normalization/moments/varianceMean7actor/batch_normalization/moments/SquaredDifference:z:0Eactor/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2,
*actor/batch_normalization/moments/varianceÎ
)actor/batch_normalization/moments/SqueezeSqueeze/actor/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)actor/batch_normalization/moments/SqueezeÖ
+actor/batch_normalization/moments/Squeeze_1Squeeze3actor/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2-
+actor/batch_normalization/moments/Squeeze_1§
/actor/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/actor/batch_normalization/AssignMovingAvg/decayò
8actor/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpAactor_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02:
8actor/batch_normalization/AssignMovingAvg/ReadVariableOp
-actor/batch_normalization/AssignMovingAvg/subSub@actor/batch_normalization/AssignMovingAvg/ReadVariableOp:value:02actor/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:2/
-actor/batch_normalization/AssignMovingAvg/sub÷
-actor/batch_normalization/AssignMovingAvg/mulMul1actor/batch_normalization/AssignMovingAvg/sub:z:08actor/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2/
-actor/batch_normalization/AssignMovingAvg/mulÁ
)actor/batch_normalization/AssignMovingAvgAssignSubVariableOpAactor_batch_normalization_assignmovingavg_readvariableop_resource1actor/batch_normalization/AssignMovingAvg/mul:z:09^actor/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02+
)actor/batch_normalization/AssignMovingAvg«
1actor/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization/AssignMovingAvg_1/decayø
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpCactor_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp
/actor/batch_normalization/AssignMovingAvg_1/subSubBactor/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:04actor/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:21
/actor/batch_normalization/AssignMovingAvg_1/subÿ
/actor/batch_normalization/AssignMovingAvg_1/mulMul3actor/batch_normalization/AssignMovingAvg_1/sub:z:0:actor/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:21
/actor/batch_normalization/AssignMovingAvg_1/mulË
+actor/batch_normalization/AssignMovingAvg_1AssignSubVariableOpCactor_batch_normalization_assignmovingavg_1_readvariableop_resource3actor/batch_normalization/AssignMovingAvg_1/mul:z:0;^actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization/AssignMovingAvg_1
)actor/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)actor/batch_normalization/batchnorm/add/yê
'actor/batch_normalization/batchnorm/addAddV24actor/batch_normalization/moments/Squeeze_1:output:02actor/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/add±
)actor/batch_normalization/batchnorm/RsqrtRsqrt+actor/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/Rsqrtì
6actor/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?actor_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype028
6actor/batch_normalization/batchnorm/mul/ReadVariableOpí
'actor/batch_normalization/batchnorm/mulMul-actor/batch_normalization/batchnorm/Rsqrt:y:0>actor/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/mulÅ
)actor/batch_normalization/batchnorm/mul_1Mulinput_1+actor/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/mul_1ã
)actor/batch_normalization/batchnorm/mul_2Mul2actor/batch_normalization/moments/Squeeze:output:0+actor/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/mul_2à
2actor/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;actor_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype024
2actor/batch_normalization/batchnorm/ReadVariableOpé
'actor/batch_normalization/batchnorm/subSub:actor/batch_normalization/batchnorm/ReadVariableOp:value:0-actor/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/subí
)actor/batch_normalization/batchnorm/add_1AddV2-actor/batch_normalization/batchnorm/mul_1:z:0+actor/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/add_1²
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!actor/dense/MatMul/ReadVariableOp¿
actor/dense/MatMulMatMul-actor/batch_normalization/batchnorm/add_1:z:0)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/MatMul±
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp²
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/ReluÂ
:actor/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:actor/batch_normalization_1/moments/mean/reduction_indicesü
(actor/batch_normalization_1/moments/meanMeanactor/dense/Relu:activations:0Cactor/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2*
(actor/batch_normalization_1/moments/meanÑ
0actor/batch_normalization_1/moments/StopGradientStopGradient1actor/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	22
0actor/batch_normalization_1/moments/StopGradient
5actor/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceactor/dense/Relu:activations:09actor/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5actor/batch_normalization_1/moments/SquaredDifferenceÊ
>actor/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>actor/batch_normalization_1/moments/variance/reduction_indices£
,actor/batch_normalization_1/moments/varianceMean9actor/batch_normalization_1/moments/SquaredDifference:z:0Gactor/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2.
,actor/batch_normalization_1/moments/varianceÕ
+actor/batch_normalization_1/moments/SqueezeSqueeze1actor/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2-
+actor/batch_normalization_1/moments/SqueezeÝ
-actor/batch_normalization_1/moments/Squeeze_1Squeeze5actor/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2/
-actor/batch_normalization_1/moments/Squeeze_1«
1actor/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization_1/AssignMovingAvg/decayù
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpCactor_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02<
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp
/actor/batch_normalization_1/AssignMovingAvg/subSubBactor/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:04actor/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_1/AssignMovingAvg/sub
/actor/batch_normalization_1/AssignMovingAvg/mulMul3actor/batch_normalization_1/AssignMovingAvg/sub:z:0:actor/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_1/AssignMovingAvg/mulË
+actor/batch_normalization_1/AssignMovingAvgAssignSubVariableOpCactor_batch_normalization_1_assignmovingavg_readvariableop_resource3actor/batch_normalization_1/AssignMovingAvg/mul:z:0;^actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization_1/AssignMovingAvg¯
3actor/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<25
3actor/batch_normalization_1/AssignMovingAvg_1/decayÿ
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpEactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02>
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp
1actor/batch_normalization_1/AssignMovingAvg_1/subSubDactor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:06actor/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_1/AssignMovingAvg_1/sub
1actor/batch_normalization_1/AssignMovingAvg_1/mulMul5actor/batch_normalization_1/AssignMovingAvg_1/sub:z:0<actor/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_1/AssignMovingAvg_1/mulÕ
-actor/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpEactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource5actor/batch_normalization_1/AssignMovingAvg_1/mul:z:0=^actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02/
-actor/batch_normalization_1/AssignMovingAvg_1
+actor/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_1/batchnorm/add/yó
)actor/batch_normalization_1/batchnorm/addAddV26actor/batch_normalization_1/moments/Squeeze_1:output:04actor/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/add¸
+actor/batch_normalization_1/batchnorm/RsqrtRsqrt-actor/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/Rsqrtó
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_1/batchnorm/mulMul/actor/batch_normalization_1/batchnorm/Rsqrt:y:0@actor/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/mulã
+actor/batch_normalization_1/batchnorm/mul_1Mulactor/dense/Relu:activations:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/mul_1ì
+actor/batch_normalization_1/batchnorm/mul_2Mul4actor/batch_normalization_1/moments/Squeeze:output:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/mul_2ç
4actor/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_1/batchnorm/ReadVariableOpò
)actor/batch_normalization_1/batchnorm/subSub<actor/batch_normalization_1/batchnorm/ReadVariableOp:value:0/actor/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/subö
+actor/batch_normalization_1/batchnorm/add_1AddV2/actor/batch_normalization_1/batchnorm/mul_1:z:0-actor/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/add_1¹
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#actor/dense_1/MatMul/ReadVariableOpÇ
actor/dense_1/MatMulMatMul/actor/batch_normalization_1/batchnorm/add_1:z:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/MatMul·
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOpº
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/BiasAdd
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/ReluÂ
:actor/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:actor/batch_normalization_2/moments/mean/reduction_indicesþ
(actor/batch_normalization_2/moments/meanMean actor/dense_1/Relu:activations:0Cactor/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2*
(actor/batch_normalization_2/moments/meanÑ
0actor/batch_normalization_2/moments/StopGradientStopGradient1actor/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	22
0actor/batch_normalization_2/moments/StopGradient
5actor/batch_normalization_2/moments/SquaredDifferenceSquaredDifference actor/dense_1/Relu:activations:09actor/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5actor/batch_normalization_2/moments/SquaredDifferenceÊ
>actor/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>actor/batch_normalization_2/moments/variance/reduction_indices£
,actor/batch_normalization_2/moments/varianceMean9actor/batch_normalization_2/moments/SquaredDifference:z:0Gactor/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2.
,actor/batch_normalization_2/moments/varianceÕ
+actor/batch_normalization_2/moments/SqueezeSqueeze1actor/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2-
+actor/batch_normalization_2/moments/SqueezeÝ
-actor/batch_normalization_2/moments/Squeeze_1Squeeze5actor/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2/
-actor/batch_normalization_2/moments/Squeeze_1«
1actor/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization_2/AssignMovingAvg/decayù
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpCactor_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02<
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp
/actor/batch_normalization_2/AssignMovingAvg/subSubBactor/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:04actor/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_2/AssignMovingAvg/sub
/actor/batch_normalization_2/AssignMovingAvg/mulMul3actor/batch_normalization_2/AssignMovingAvg/sub:z:0:actor/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_2/AssignMovingAvg/mulË
+actor/batch_normalization_2/AssignMovingAvgAssignSubVariableOpCactor_batch_normalization_2_assignmovingavg_readvariableop_resource3actor/batch_normalization_2/AssignMovingAvg/mul:z:0;^actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization_2/AssignMovingAvg¯
3actor/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<25
3actor/batch_normalization_2/AssignMovingAvg_1/decayÿ
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpEactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02>
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp
1actor/batch_normalization_2/AssignMovingAvg_1/subSubDactor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:06actor/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_2/AssignMovingAvg_1/sub
1actor/batch_normalization_2/AssignMovingAvg_1/mulMul5actor/batch_normalization_2/AssignMovingAvg_1/sub:z:0<actor/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_2/AssignMovingAvg_1/mulÕ
-actor/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpEactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource5actor/batch_normalization_2/AssignMovingAvg_1/mul:z:0=^actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02/
-actor/batch_normalization_2/AssignMovingAvg_1
+actor/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_2/batchnorm/add/yó
)actor/batch_normalization_2/batchnorm/addAddV26actor/batch_normalization_2/moments/Squeeze_1:output:04actor/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/add¸
+actor/batch_normalization_2/batchnorm/RsqrtRsqrt-actor/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/Rsqrtó
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_2/batchnorm/mulMul/actor/batch_normalization_2/batchnorm/Rsqrt:y:0@actor/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/mulå
+actor/batch_normalization_2/batchnorm/mul_1Mul actor/dense_1/Relu:activations:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/mul_1ì
+actor/batch_normalization_2/batchnorm/mul_2Mul4actor/batch_normalization_2/moments/Squeeze:output:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/mul_2ç
4actor/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_2/batchnorm/ReadVariableOpò
)actor/batch_normalization_2/batchnorm/subSub<actor/batch_normalization_2/batchnorm/ReadVariableOp:value:0/actor/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/subö
+actor/batch_normalization_2/batchnorm/add_1AddV2/actor/batch_normalization_2/batchnorm/mul_1:z:0-actor/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/add_1
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
mean/MatMul/ReadVariableOp«
mean/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/MatMul
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/BiasAdd²
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!log_std_dev/MatMul/ReadVariableOpÀ
log_std_dev/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/MatMul°
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp±
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y«
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2À

NoOpNoOp*^actor/batch_normalization/AssignMovingAvg9^actor/batch_normalization/AssignMovingAvg/ReadVariableOp,^actor/batch_normalization/AssignMovingAvg_1;^actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^actor/batch_normalization/batchnorm/ReadVariableOp7^actor/batch_normalization/batchnorm/mul/ReadVariableOp,^actor/batch_normalization_1/AssignMovingAvg;^actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp.^actor/batch_normalization_1/AssignMovingAvg_1=^actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^actor/batch_normalization_1/batchnorm/ReadVariableOp9^actor/batch_normalization_1/batchnorm/mul/ReadVariableOp,^actor/batch_normalization_2/AssignMovingAvg;^actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp.^actor/batch_normalization_2/AssignMovingAvg_1=^actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^actor/batch_normalization_2/batchnorm/ReadVariableOp9^actor/batch_normalization_2/batchnorm/mul/ReadVariableOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2V
)actor/batch_normalization/AssignMovingAvg)actor/batch_normalization/AssignMovingAvg2t
8actor/batch_normalization/AssignMovingAvg/ReadVariableOp8actor/batch_normalization/AssignMovingAvg/ReadVariableOp2Z
+actor/batch_normalization/AssignMovingAvg_1+actor/batch_normalization/AssignMovingAvg_12x
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2actor/batch_normalization/batchnorm/ReadVariableOp2actor/batch_normalization/batchnorm/ReadVariableOp2p
6actor/batch_normalization/batchnorm/mul/ReadVariableOp6actor/batch_normalization/batchnorm/mul/ReadVariableOp2Z
+actor/batch_normalization_1/AssignMovingAvg+actor/batch_normalization_1/AssignMovingAvg2x
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^
-actor/batch_normalization_1/AssignMovingAvg_1-actor/batch_normalization_1/AssignMovingAvg_12|
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4actor/batch_normalization_1/batchnorm/ReadVariableOp4actor/batch_normalization_1/batchnorm/ReadVariableOp2t
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp2Z
+actor/batch_normalization_2/AssignMovingAvg+actor/batch_normalization_2/AssignMovingAvg2x
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^
-actor/batch_normalization_2/AssignMovingAvg_1-actor/batch_normalization_2/AssignMovingAvg_12|
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4actor/batch_normalization_2/batchnorm/ReadVariableOp4actor/batch_normalization_2/batchnorm/ReadVariableOp2t
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ
°
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26716816

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Âj

C__inference_actor_layer_call_and_return_conditional_losses_26719157

inputsC
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	F
7batch_normalization_1_batchnorm_readvariableop_resource:	J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	F
7batch_normalization_2_batchnorm_readvariableop_resource:	J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_2_batchnorm_readvariableop_1_resource:	H
9batch_normalization_2_batchnorm_readvariableop_2_resource:	
identity¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢0batch_normalization_1/batchnorm/ReadVariableOp_1¢0batch_normalization_1/batchnorm/ReadVariableOp_2¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢0batch_normalization_2/batchnorm/ReadVariableOp_1¢0batch_normalization_2/batchnorm/ReadVariableOp_2¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOpÎ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yØ
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul²
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/mul_1Ô
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Õ
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2Ô
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2Ó
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/add_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp§
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/ReluÕ
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yá
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_1/batchnorm/Rsqrtá
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/mulË
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_1/batchnorm/mul_1Û
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1Þ
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_1/batchnorm/mul_2Û
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2Ü
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_1/batchnorm/add_1§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¯
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/ReluÕ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yá
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/add¦
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_2/batchnorm/Rsqrtá
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/mulÍ
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_2/batchnorm/mul_1Û
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Þ
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_2/batchnorm/mul_2Û
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2Ü
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/subÞ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_2/batchnorm/add_1
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
øî
®
#__inference__wrapped_model_26716792
input_1\
Nprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_resource:`
Rprivate__mlp_actor_actor_batch_normalization_batchnorm_mul_readvariableop_resource:^
Pprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_1_resource:^
Pprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_2_resource:P
=private__mlp_actor_actor_dense_matmul_readvariableop_resource:	M
>private__mlp_actor_actor_dense_biasadd_readvariableop_resource:	_
Pprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_resource:	c
Tprivate__mlp_actor_actor_batch_normalization_1_batchnorm_mul_readvariableop_resource:	a
Rprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_1_resource:	a
Rprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_2_resource:	S
?private__mlp_actor_actor_dense_1_matmul_readvariableop_resource:
O
@private__mlp_actor_actor_dense_1_biasadd_readvariableop_resource:	_
Pprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_resource:	c
Tprivate__mlp_actor_actor_batch_normalization_2_batchnorm_mul_readvariableop_resource:	a
Rprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_1_resource:	a
Rprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_2_resource:	I
6private__mlp_actor_mean_matmul_readvariableop_resource:	E
7private__mlp_actor_mean_biasadd_readvariableop_resource:P
=private__mlp_actor_log_std_dev_matmul_readvariableop_resource:	L
>private__mlp_actor_log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢Eprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp¢Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1¢Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2¢Iprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOp¢Gprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp¢Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1¢Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2¢Kprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOp¢Gprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp¢Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1¢Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2¢Kprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOp¢5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp¢4private__mlp_actor/actor/dense/MatMul/ReadVariableOp¢7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp¢6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp¢5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp¢4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp¢.private__mlp_actor/mean/BiasAdd/ReadVariableOp¢-private__mlp_actor/mean/MatMul/ReadVariableOp
Eprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOpReadVariableOpNprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02G
Eprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOpÁ
<private__mlp_actor/actor/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2>
<private__mlp_actor/actor/batch_normalization/batchnorm/add/y¼
:private__mlp_actor/actor/batch_normalization/batchnorm/addAddV2Mprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp:value:0Eprivate__mlp_actor/actor/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:private__mlp_actor/actor/batch_normalization/batchnorm/addê
<private__mlp_actor/actor/batch_normalization/batchnorm/RsqrtRsqrt>private__mlp_actor/actor/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<private__mlp_actor/actor/batch_normalization/batchnorm/Rsqrt¥
Iprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpRprivate__mlp_actor_actor_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02K
Iprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOp¹
:private__mlp_actor/actor/batch_normalization/batchnorm/mulMul@private__mlp_actor/actor/batch_normalization/batchnorm/Rsqrt:y:0Qprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:private__mlp_actor/actor/batch_normalization/batchnorm/mulþ
<private__mlp_actor/actor/batch_normalization/batchnorm/mul_1Mulinput_1>private__mlp_actor/actor/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<private__mlp_actor/actor/batch_normalization/batchnorm/mul_1
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpPprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1¹
<private__mlp_actor/actor/batch_normalization/batchnorm/mul_2MulOprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1:value:0>private__mlp_actor/actor/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<private__mlp_actor/actor/batch_normalization/batchnorm/mul_2
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpPprivate__mlp_actor_actor_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02I
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2·
:private__mlp_actor/actor/batch_normalization/batchnorm/subSubOprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2:value:0@private__mlp_actor/actor/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:private__mlp_actor/actor/batch_normalization/batchnorm/sub¹
<private__mlp_actor/actor/batch_normalization/batchnorm/add_1AddV2@private__mlp_actor/actor/batch_normalization/batchnorm/mul_1:z:0>private__mlp_actor/actor/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<private__mlp_actor/actor/batch_normalization/batchnorm/add_1ë
4private__mlp_actor/actor/dense/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_actor_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4private__mlp_actor/actor/dense/MatMul/ReadVariableOp
%private__mlp_actor/actor/dense/MatMulMatMul@private__mlp_actor/actor/batch_normalization/batchnorm/add_1:z:0<private__mlp_actor/actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%private__mlp_actor/actor/dense/MatMulê
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOpReadVariableOp>private__mlp_actor_actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5private__mlp_actor/actor/dense/BiasAdd/ReadVariableOpþ
&private__mlp_actor/actor/dense/BiasAddBiasAdd/private__mlp_actor/actor/dense/MatMul:product:0=private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&private__mlp_actor/actor/dense/BiasAdd¶
#private__mlp_actor/actor/dense/ReluRelu/private__mlp_actor/actor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#private__mlp_actor/actor/dense/Relu 
Gprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpPprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02I
Gprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOpÅ
>private__mlp_actor/actor/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2@
>private__mlp_actor/actor/batch_normalization_1/batchnorm/add/yÅ
<private__mlp_actor/actor/batch_normalization_1/batchnorm/addAddV2Oprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp:value:0Gprivate__mlp_actor/actor/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_1/batchnorm/addñ
>private__mlp_actor/actor/batch_normalization_1/batchnorm/RsqrtRsqrt@private__mlp_actor/actor/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2@
>private__mlp_actor/actor/batch_normalization_1/batchnorm/Rsqrt¬
Kprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpTprivate__mlp_actor_actor_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02M
Kprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOpÂ
<private__mlp_actor/actor/batch_normalization_1/batchnorm/mulMulBprivate__mlp_actor/actor/batch_normalization_1/batchnorm/Rsqrt:y:0Sprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_1/batchnorm/mul¯
>private__mlp_actor/actor/batch_normalization_1/batchnorm/mul_1Mul1private__mlp_actor/actor/dense/Relu:activations:0@private__mlp_actor/actor/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>private__mlp_actor/actor/batch_normalization_1/batchnorm/mul_1¦
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpRprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02K
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1Â
>private__mlp_actor/actor/batch_normalization_1/batchnorm/mul_2MulQprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0@private__mlp_actor/actor/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2@
>private__mlp_actor/actor/batch_normalization_1/batchnorm/mul_2¦
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpRprivate__mlp_actor_actor_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02K
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2À
<private__mlp_actor/actor/batch_normalization_1/batchnorm/subSubQprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Bprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_1/batchnorm/subÂ
>private__mlp_actor/actor/batch_normalization_1/batchnorm/add_1AddV2Bprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul_1:z:0@private__mlp_actor/actor/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>private__mlp_actor/actor/batch_normalization_1/batchnorm/add_1ò
6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOpReadVariableOp?private__mlp_actor_actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype028
6private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp
'private__mlp_actor/actor/dense_1/MatMulMatMulBprivate__mlp_actor/actor/batch_normalization_1/batchnorm/add_1:z:0>private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'private__mlp_actor/actor/dense_1/MatMulð
7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp@private__mlp_actor_actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp
(private__mlp_actor/actor/dense_1/BiasAddBiasAdd1private__mlp_actor/actor/dense_1/MatMul:product:0?private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(private__mlp_actor/actor/dense_1/BiasAdd¼
%private__mlp_actor/actor/dense_1/ReluRelu1private__mlp_actor/actor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%private__mlp_actor/actor/dense_1/Relu 
Gprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpPprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02I
Gprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOpÅ
>private__mlp_actor/actor/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2@
>private__mlp_actor/actor/batch_normalization_2/batchnorm/add/yÅ
<private__mlp_actor/actor/batch_normalization_2/batchnorm/addAddV2Oprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp:value:0Gprivate__mlp_actor/actor/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_2/batchnorm/addñ
>private__mlp_actor/actor/batch_normalization_2/batchnorm/RsqrtRsqrt@private__mlp_actor/actor/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2@
>private__mlp_actor/actor/batch_normalization_2/batchnorm/Rsqrt¬
Kprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpTprivate__mlp_actor_actor_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02M
Kprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOpÂ
<private__mlp_actor/actor/batch_normalization_2/batchnorm/mulMulBprivate__mlp_actor/actor/batch_normalization_2/batchnorm/Rsqrt:y:0Sprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_2/batchnorm/mul±
>private__mlp_actor/actor/batch_normalization_2/batchnorm/mul_1Mul3private__mlp_actor/actor/dense_1/Relu:activations:0@private__mlp_actor/actor/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>private__mlp_actor/actor/batch_normalization_2/batchnorm/mul_1¦
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpRprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02K
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1Â
>private__mlp_actor/actor/batch_normalization_2/batchnorm/mul_2MulQprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0@private__mlp_actor/actor/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2@
>private__mlp_actor/actor/batch_normalization_2/batchnorm/mul_2¦
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpRprivate__mlp_actor_actor_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02K
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2À
<private__mlp_actor/actor/batch_normalization_2/batchnorm/subSubQprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Bprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2>
<private__mlp_actor/actor/batch_normalization_2/batchnorm/subÂ
>private__mlp_actor/actor/batch_normalization_2/batchnorm/add_1AddV2Bprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul_1:z:0@private__mlp_actor/actor/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>private__mlp_actor/actor/batch_normalization_2/batchnorm/add_1Ö
-private__mlp_actor/mean/MatMul/ReadVariableOpReadVariableOp6private__mlp_actor_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-private__mlp_actor/mean/MatMul/ReadVariableOp÷
private__mlp_actor/mean/MatMulMatMulBprivate__mlp_actor/actor/batch_normalization_2/batchnorm/add_1:z:05private__mlp_actor/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
private__mlp_actor/mean/MatMulÔ
.private__mlp_actor/mean/BiasAdd/ReadVariableOpReadVariableOp7private__mlp_actor_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.private__mlp_actor/mean/BiasAdd/ReadVariableOpá
private__mlp_actor/mean/BiasAddBiasAdd(private__mlp_actor/mean/MatMul:product:06private__mlp_actor/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
private__mlp_actor/mean/BiasAddë
4private__mlp_actor/log_std_dev/MatMul/ReadVariableOpReadVariableOp=private__mlp_actor_log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4private__mlp_actor/log_std_dev/MatMul/ReadVariableOp
%private__mlp_actor/log_std_dev/MatMulMatMulBprivate__mlp_actor/actor/batch_normalization_2/batchnorm/add_1:z:0<private__mlp_actor/log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%private__mlp_actor/log_std_dev/MatMulé
5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOpReadVariableOp>private__mlp_actor_log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOpý
&private__mlp_actor/log_std_dev/BiasAddBiasAdd/private__mlp_actor/log_std_dev/MatMul:product:0=private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&private__mlp_actor/log_std_dev/BiasAdd
*private__mlp_actor/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2,
*private__mlp_actor/clip_by_value/Minimum/y÷
(private__mlp_actor/clip_by_value/MinimumMinimum/private__mlp_actor/log_std_dev/BiasAdd:output:03private__mlp_actor/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(private__mlp_actor/clip_by_value/Minimum
"private__mlp_actor/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2$
"private__mlp_actor/clip_by_value/yÜ
 private__mlp_actor/clip_by_valueMaximum,private__mlp_actor/clip_by_value/Minimum:z:0+private__mlp_actor/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 private__mlp_actor/clip_by_value
private__mlp_actor/ExpExp$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Exp
private__mlp_actor/ShapeShape(private__mlp_actor/mean/BiasAdd:output:0*
T0*
_output_shapes
:2
private__mlp_actor/Shape
%private__mlp_actor/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%private__mlp_actor/random_normal/mean
'private__mlp_actor/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'private__mlp_actor/random_normal/stddevô
5private__mlp_actor/random_normal/RandomStandardNormalRandomStandardNormal!private__mlp_actor/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed27
5private__mlp_actor/random_normal/RandomStandardNormal÷
$private__mlp_actor/random_normal/mulMul>private__mlp_actor/random_normal/RandomStandardNormal:output:00private__mlp_actor/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$private__mlp_actor/random_normal/mulÙ
 private__mlp_actor/random_normalAddV2(private__mlp_actor/random_normal/mul:z:0.private__mlp_actor/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 private__mlp_actor/random_normal«
private__mlp_actor/mulMul$private__mlp_actor/random_normal:z:0private__mlp_actor/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul±
private__mlp_actor/addAddV2(private__mlp_actor/mean/BiasAdd:output:0private__mlp_actor/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/add¯
private__mlp_actor/subSubprivate__mlp_actor/add:z:0(private__mlp_actor/mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/sub
private__mlp_actor/Exp_1Exp$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Exp_1}
private__mlp_actor/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
private__mlp_actor/add_1/y²
private__mlp_actor/add_1AddV2private__mlp_actor/Exp_1:y:0#private__mlp_actor/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/add_1¯
private__mlp_actor/truedivRealDivprivate__mlp_actor/sub:z:0private__mlp_actor/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/truedivy
private__mlp_actor/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/pow/y¬
private__mlp_actor/powPowprivate__mlp_actor/truediv:z:0!private__mlp_actor/pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/pow}
private__mlp_actor/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/mul_1/x¸
private__mlp_actor/mul_1Mul#private__mlp_actor/mul_1/x:output:0$private__mlp_actor/clip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_1©
private__mlp_actor/add_2AddV2private__mlp_actor/pow:z:0private__mlp_actor/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/add_2}
private__mlp_actor/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2
private__mlp_actor/add_3/y²
private__mlp_actor/add_3AddV2private__mlp_actor/add_2:z:0#private__mlp_actor/add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/add_3}
private__mlp_actor/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2
private__mlp_actor/mul_2/x°
private__mlp_actor/mul_2Mul#private__mlp_actor/mul_2/x:output:0private__mlp_actor/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_2
(private__mlp_actor/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(private__mlp_actor/Sum/reduction_indices¶
private__mlp_actor/SumSumprivate__mlp_actor/mul_2:z:01private__mlp_actor/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Sum}
private__mlp_actor/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2
private__mlp_actor/sub_1/x®
private__mlp_actor/sub_1Sub#private__mlp_actor/sub_1/x:output:0private__mlp_actor/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/sub_1}
private__mlp_actor/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2
private__mlp_actor/mul_3/x®
private__mlp_actor/mul_3Mul#private__mlp_actor/mul_3/x:output:0private__mlp_actor/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_3
private__mlp_actor/SoftplusSoftplusprivate__mlp_actor/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Softplus¶
private__mlp_actor/sub_2Subprivate__mlp_actor/sub_1:z:0)private__mlp_actor/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/sub_2}
private__mlp_actor/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
private__mlp_actor/mul_4/x°
private__mlp_actor/mul_4Mul#private__mlp_actor/mul_4/x:output:0private__mlp_actor/sub_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_4
*private__mlp_actor/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*private__mlp_actor/Sum_1/reduction_indices¼
private__mlp_actor/Sum_1Sumprivate__mlp_actor/mul_4:z:03private__mlp_actor/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Sum_1­
private__mlp_actor/sub_3Subprivate__mlp_actor/Sum:output:0!private__mlp_actor/Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/sub_3
private__mlp_actor/TanhTanh(private__mlp_actor/mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Tanh
private__mlp_actor/Tanh_1Tanhprivate__mlp_actor/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/Tanh_1}
private__mlp_actor/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
private__mlp_actor/mul_5/y¯
private__mlp_actor/mul_5Mulprivate__mlp_actor/Tanh:y:0#private__mlp_actor/mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_5}
private__mlp_actor/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
private__mlp_actor/mul_6/y±
private__mlp_actor/mul_6Mulprivate__mlp_actor/Tanh_1:y:0#private__mlp_actor/mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
private__mlp_actor/mul_6w
IdentityIdentityprivate__mlp_actor/mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity{

Identity_1Identityprivate__mlp_actor/mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1w

Identity_2Identityprivate__mlp_actor/sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOpF^private__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOpH^private__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1H^private__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2J^private__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOpH^private__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOpJ^private__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1J^private__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2L^private__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOpH^private__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOpJ^private__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1J^private__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2L^private__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOp6^private__mlp_actor/actor/dense/BiasAdd/ReadVariableOp5^private__mlp_actor/actor/dense/MatMul/ReadVariableOp8^private__mlp_actor/actor/dense_1/BiasAdd/ReadVariableOp7^private__mlp_actor/actor/dense_1/MatMul/ReadVariableOp6^private__mlp_actor/log_std_dev/BiasAdd/ReadVariableOp5^private__mlp_actor/log_std_dev/MatMul/ReadVariableOp/^private__mlp_actor/mean/BiasAdd/ReadVariableOp.^private__mlp_actor/mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2
Eprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOpEprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp2
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_1Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_12
Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_2Gprivate__mlp_actor/actor/batch_normalization/batchnorm/ReadVariableOp_22
Iprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOpIprivate__mlp_actor/actor/batch_normalization/batchnorm/mul/ReadVariableOp2
Gprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOpGprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp2
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_1Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_12
Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_2Iprivate__mlp_actor/actor/batch_normalization_1/batchnorm/ReadVariableOp_22
Kprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOpKprivate__mlp_actor/actor/batch_normalization_1/batchnorm/mul/ReadVariableOp2
Gprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOpGprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp2
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_1Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_12
Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_2Iprivate__mlp_actor/actor/batch_normalization_2/batchnorm/ReadVariableOp_22
Kprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOpKprivate__mlp_actor/actor/batch_normalization_2/batchnorm/mul/ReadVariableOp2n
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Í2
Ñ	
!__inference__traced_save_26719742
file_prefix*
&savev2_mean_kernel_read_readvariableop(
$savev2_mean_bias_read_readvariableop1
-savev2_log_std_dev_kernel_read_readvariableop/
+savev2_log_std_dev_bias_read_readvariableop8
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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename©
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*»
value±B®B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesá	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_mean_kernel_read_readvariableop$savev2_mean_bias_read_readvariableop-savev2_log_std_dev_kernel_read_readvariableop+savev2_log_std_dev_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*°
_input_shapes
: :	::	::::	::::
:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::

_output_shapes
: 

©
5__inference_private__mlp_actor_layer_call_fn_26719091
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:	

unknown_18:
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_267179702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¨

ô
B__inference_mean_layer_call_and_return_conditional_losses_26719349

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×®
Ý
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718433
xI
;actor_batch_normalization_batchnorm_readvariableop_resource:M
?actor_batch_normalization_batchnorm_mul_readvariableop_resource:K
=actor_batch_normalization_batchnorm_readvariableop_1_resource:K
=actor_batch_normalization_batchnorm_readvariableop_2_resource:=
*actor_dense_matmul_readvariableop_resource:	:
+actor_dense_biasadd_readvariableop_resource:	L
=actor_batch_normalization_1_batchnorm_readvariableop_resource:	P
Aactor_batch_normalization_1_batchnorm_mul_readvariableop_resource:	N
?actor_batch_normalization_1_batchnorm_readvariableop_1_resource:	N
?actor_batch_normalization_1_batchnorm_readvariableop_2_resource:	@
,actor_dense_1_matmul_readvariableop_resource:
<
-actor_dense_1_biasadd_readvariableop_resource:	L
=actor_batch_normalization_2_batchnorm_readvariableop_resource:	P
Aactor_batch_normalization_2_batchnorm_mul_readvariableop_resource:	N
?actor_batch_normalization_2_batchnorm_readvariableop_1_resource:	N
?actor_batch_normalization_2_batchnorm_readvariableop_2_resource:	6
#mean_matmul_readvariableop_resource:	2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢2actor/batch_normalization/batchnorm/ReadVariableOp¢4actor/batch_normalization/batchnorm/ReadVariableOp_1¢4actor/batch_normalization/batchnorm/ReadVariableOp_2¢6actor/batch_normalization/batchnorm/mul/ReadVariableOp¢4actor/batch_normalization_1/batchnorm/ReadVariableOp¢6actor/batch_normalization_1/batchnorm/ReadVariableOp_1¢6actor/batch_normalization_1/batchnorm/ReadVariableOp_2¢8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp¢4actor/batch_normalization_2/batchnorm/ReadVariableOp¢6actor/batch_normalization_2/batchnorm/ReadVariableOp_1¢6actor/batch_normalization_2/batchnorm/ReadVariableOp_2¢8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"actor/dense/BiasAdd/ReadVariableOp¢!actor/dense/MatMul/ReadVariableOp¢$actor/dense_1/BiasAdd/ReadVariableOp¢#actor/dense_1/MatMul/ReadVariableOp¢"log_std_dev/BiasAdd/ReadVariableOp¢!log_std_dev/MatMul/ReadVariableOp¢mean/BiasAdd/ReadVariableOp¢mean/MatMul/ReadVariableOpà
2actor/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;actor_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype024
2actor/batch_normalization/batchnorm/ReadVariableOp
)actor/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)actor/batch_normalization/batchnorm/add/yð
'actor/batch_normalization/batchnorm/addAddV2:actor/batch_normalization/batchnorm/ReadVariableOp:value:02actor/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/add±
)actor/batch_normalization/batchnorm/RsqrtRsqrt+actor/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/Rsqrtì
6actor/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?actor_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype028
6actor/batch_normalization/batchnorm/mul/ReadVariableOpí
'actor/batch_normalization/batchnorm/mulMul-actor/batch_normalization/batchnorm/Rsqrt:y:0>actor/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/mul¿
)actor/batch_normalization/batchnorm/mul_1Mulx+actor/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/mul_1æ
4actor/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=actor_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype026
4actor/batch_normalization/batchnorm/ReadVariableOp_1í
)actor/batch_normalization/batchnorm/mul_2Mul<actor/batch_normalization/batchnorm/ReadVariableOp_1:value:0+actor/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/mul_2æ
4actor/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=actor_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype026
4actor/batch_normalization/batchnorm/ReadVariableOp_2ë
'actor/batch_normalization/batchnorm/subSub<actor/batch_normalization/batchnorm/ReadVariableOp_2:value:0-actor/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/subí
)actor/batch_normalization/batchnorm/add_1AddV2-actor/batch_normalization/batchnorm/mul_1:z:0+actor/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/add_1²
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!actor/dense/MatMul/ReadVariableOp¿
actor/dense/MatMulMatMul-actor/batch_normalization/batchnorm/add_1:z:0)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/MatMul±
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp²
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/Reluç
4actor/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_1/batchnorm/ReadVariableOp
+actor/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_1/batchnorm/add/yù
)actor/batch_normalization_1/batchnorm/addAddV2<actor/batch_normalization_1/batchnorm/ReadVariableOp:value:04actor/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/add¸
+actor/batch_normalization_1/batchnorm/RsqrtRsqrt-actor/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/Rsqrtó
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_1/batchnorm/mulMul/actor/batch_normalization_1/batchnorm/Rsqrt:y:0@actor/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/mulã
+actor/batch_normalization_1/batchnorm/mul_1Mulactor/dense/Relu:activations:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/mul_1í
6actor/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?actor_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_1/batchnorm/ReadVariableOp_1ö
+actor/batch_normalization_1/batchnorm/mul_2Mul>actor/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/mul_2í
6actor/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?actor_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_1/batchnorm/ReadVariableOp_2ô
)actor/batch_normalization_1/batchnorm/subSub>actor/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/actor/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/subö
+actor/batch_normalization_1/batchnorm/add_1AddV2/actor/batch_normalization_1/batchnorm/mul_1:z:0-actor/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/add_1¹
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#actor/dense_1/MatMul/ReadVariableOpÇ
actor/dense_1/MatMulMatMul/actor/batch_normalization_1/batchnorm/add_1:z:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/MatMul·
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOpº
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/BiasAdd
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/Reluç
4actor/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_2/batchnorm/ReadVariableOp
+actor/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_2/batchnorm/add/yù
)actor/batch_normalization_2/batchnorm/addAddV2<actor/batch_normalization_2/batchnorm/ReadVariableOp:value:04actor/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/add¸
+actor/batch_normalization_2/batchnorm/RsqrtRsqrt-actor/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/Rsqrtó
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_2/batchnorm/mulMul/actor/batch_normalization_2/batchnorm/Rsqrt:y:0@actor/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/mulå
+actor/batch_normalization_2/batchnorm/mul_1Mul actor/dense_1/Relu:activations:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/mul_1í
6actor/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?actor_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_2/batchnorm/ReadVariableOp_1ö
+actor/batch_normalization_2/batchnorm/mul_2Mul>actor/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/mul_2í
6actor/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?actor_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_2/batchnorm/ReadVariableOp_2ô
)actor/batch_normalization_2/batchnorm/subSub>actor/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/actor/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/subö
+actor/batch_normalization_2/batchnorm/add_1AddV2/actor/batch_normalization_2/batchnorm/mul_1:z:0-actor/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/add_1
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
mean/MatMul/ReadVariableOp«
mean/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/MatMul
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/BiasAdd²
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!log_std_dev/MatMul/ReadVariableOpÀ
log_std_dev/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/MatMul°
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp±
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y«
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp3^actor/batch_normalization/batchnorm/ReadVariableOp5^actor/batch_normalization/batchnorm/ReadVariableOp_15^actor/batch_normalization/batchnorm/ReadVariableOp_27^actor/batch_normalization/batchnorm/mul/ReadVariableOp5^actor/batch_normalization_1/batchnorm/ReadVariableOp7^actor/batch_normalization_1/batchnorm/ReadVariableOp_17^actor/batch_normalization_1/batchnorm/ReadVariableOp_29^actor/batch_normalization_1/batchnorm/mul/ReadVariableOp5^actor/batch_normalization_2/batchnorm/ReadVariableOp7^actor/batch_normalization_2/batchnorm/ReadVariableOp_17^actor/batch_normalization_2/batchnorm/ReadVariableOp_29^actor/batch_normalization_2/batchnorm/mul/ReadVariableOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2h
2actor/batch_normalization/batchnorm/ReadVariableOp2actor/batch_normalization/batchnorm/ReadVariableOp2l
4actor/batch_normalization/batchnorm/ReadVariableOp_14actor/batch_normalization/batchnorm/ReadVariableOp_12l
4actor/batch_normalization/batchnorm/ReadVariableOp_24actor/batch_normalization/batchnorm/ReadVariableOp_22p
6actor/batch_normalization/batchnorm/mul/ReadVariableOp6actor/batch_normalization/batchnorm/mul/ReadVariableOp2l
4actor/batch_normalization_1/batchnorm/ReadVariableOp4actor/batch_normalization_1/batchnorm/ReadVariableOp2p
6actor/batch_normalization_1/batchnorm/ReadVariableOp_16actor/batch_normalization_1/batchnorm/ReadVariableOp_12p
6actor/batch_normalization_1/batchnorm/ReadVariableOp_26actor/batch_normalization_1/batchnorm/ReadVariableOp_22t
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp2l
4actor/batch_normalization_2/batchnorm/ReadVariableOp4actor/batch_normalization_2/batchnorm/ReadVariableOp2p
6actor/batch_normalization_2/batchnorm/ReadVariableOp_16actor/batch_normalization_2/batchnorm/ReadVariableOp_12p
6actor/batch_normalization_2/batchnorm/ReadVariableOp_26actor/batch_normalization_2/batchnorm/ReadVariableOp_22t
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ê*
ð
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719531

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
¯
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718601
xO
Aactor_batch_normalization_assignmovingavg_readvariableop_resource:Q
Cactor_batch_normalization_assignmovingavg_1_readvariableop_resource:M
?actor_batch_normalization_batchnorm_mul_readvariableop_resource:I
;actor_batch_normalization_batchnorm_readvariableop_resource:=
*actor_dense_matmul_readvariableop_resource:	:
+actor_dense_biasadd_readvariableop_resource:	R
Cactor_batch_normalization_1_assignmovingavg_readvariableop_resource:	T
Eactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	P
Aactor_batch_normalization_1_batchnorm_mul_readvariableop_resource:	L
=actor_batch_normalization_1_batchnorm_readvariableop_resource:	@
,actor_dense_1_matmul_readvariableop_resource:
<
-actor_dense_1_biasadd_readvariableop_resource:	R
Cactor_batch_normalization_2_assignmovingavg_readvariableop_resource:	T
Eactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	P
Aactor_batch_normalization_2_batchnorm_mul_readvariableop_resource:	L
=actor_batch_normalization_2_batchnorm_readvariableop_resource:	6
#mean_matmul_readvariableop_resource:	2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢)actor/batch_normalization/AssignMovingAvg¢8actor/batch_normalization/AssignMovingAvg/ReadVariableOp¢+actor/batch_normalization/AssignMovingAvg_1¢:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp¢2actor/batch_normalization/batchnorm/ReadVariableOp¢6actor/batch_normalization/batchnorm/mul/ReadVariableOp¢+actor/batch_normalization_1/AssignMovingAvg¢:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp¢-actor/batch_normalization_1/AssignMovingAvg_1¢<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢4actor/batch_normalization_1/batchnorm/ReadVariableOp¢8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp¢+actor/batch_normalization_2/AssignMovingAvg¢:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp¢-actor/batch_normalization_2/AssignMovingAvg_1¢<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢4actor/batch_normalization_2/batchnorm/ReadVariableOp¢8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"actor/dense/BiasAdd/ReadVariableOp¢!actor/dense/MatMul/ReadVariableOp¢$actor/dense_1/BiasAdd/ReadVariableOp¢#actor/dense_1/MatMul/ReadVariableOp¢"log_std_dev/BiasAdd/ReadVariableOp¢!log_std_dev/MatMul/ReadVariableOp¢mean/BiasAdd/ReadVariableOp¢mean/MatMul/ReadVariableOp¾
8actor/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8actor/batch_normalization/moments/mean/reduction_indicesØ
&actor/batch_normalization/moments/meanMeanxAactor/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&actor/batch_normalization/moments/meanÊ
.actor/batch_normalization/moments/StopGradientStopGradient/actor/batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:20
.actor/batch_normalization/moments/StopGradientí
3actor/batch_normalization/moments/SquaredDifferenceSquaredDifferencex7actor/batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3actor/batch_normalization/moments/SquaredDifferenceÆ
<actor/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2>
<actor/batch_normalization/moments/variance/reduction_indices
*actor/batch_normalization/moments/varianceMean7actor/batch_normalization/moments/SquaredDifference:z:0Eactor/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2,
*actor/batch_normalization/moments/varianceÎ
)actor/batch_normalization/moments/SqueezeSqueeze/actor/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)actor/batch_normalization/moments/SqueezeÖ
+actor/batch_normalization/moments/Squeeze_1Squeeze3actor/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2-
+actor/batch_normalization/moments/Squeeze_1§
/actor/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/actor/batch_normalization/AssignMovingAvg/decayò
8actor/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpAactor_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02:
8actor/batch_normalization/AssignMovingAvg/ReadVariableOp
-actor/batch_normalization/AssignMovingAvg/subSub@actor/batch_normalization/AssignMovingAvg/ReadVariableOp:value:02actor/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:2/
-actor/batch_normalization/AssignMovingAvg/sub÷
-actor/batch_normalization/AssignMovingAvg/mulMul1actor/batch_normalization/AssignMovingAvg/sub:z:08actor/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2/
-actor/batch_normalization/AssignMovingAvg/mulÁ
)actor/batch_normalization/AssignMovingAvgAssignSubVariableOpAactor_batch_normalization_assignmovingavg_readvariableop_resource1actor/batch_normalization/AssignMovingAvg/mul:z:09^actor/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02+
)actor/batch_normalization/AssignMovingAvg«
1actor/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization/AssignMovingAvg_1/decayø
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpCactor_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp
/actor/batch_normalization/AssignMovingAvg_1/subSubBactor/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:04actor/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:21
/actor/batch_normalization/AssignMovingAvg_1/subÿ
/actor/batch_normalization/AssignMovingAvg_1/mulMul3actor/batch_normalization/AssignMovingAvg_1/sub:z:0:actor/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:21
/actor/batch_normalization/AssignMovingAvg_1/mulË
+actor/batch_normalization/AssignMovingAvg_1AssignSubVariableOpCactor_batch_normalization_assignmovingavg_1_readvariableop_resource3actor/batch_normalization/AssignMovingAvg_1/mul:z:0;^actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization/AssignMovingAvg_1
)actor/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)actor/batch_normalization/batchnorm/add/yê
'actor/batch_normalization/batchnorm/addAddV24actor/batch_normalization/moments/Squeeze_1:output:02actor/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/add±
)actor/batch_normalization/batchnorm/RsqrtRsqrt+actor/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/Rsqrtì
6actor/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?actor_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype028
6actor/batch_normalization/batchnorm/mul/ReadVariableOpí
'actor/batch_normalization/batchnorm/mulMul-actor/batch_normalization/batchnorm/Rsqrt:y:0>actor/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/mul¿
)actor/batch_normalization/batchnorm/mul_1Mulx+actor/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/mul_1ã
)actor/batch_normalization/batchnorm/mul_2Mul2actor/batch_normalization/moments/Squeeze:output:0+actor/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/mul_2à
2actor/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;actor_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype024
2actor/batch_normalization/batchnorm/ReadVariableOpé
'actor/batch_normalization/batchnorm/subSub:actor/batch_normalization/batchnorm/ReadVariableOp:value:0-actor/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/subí
)actor/batch_normalization/batchnorm/add_1AddV2-actor/batch_normalization/batchnorm/mul_1:z:0+actor/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/add_1²
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!actor/dense/MatMul/ReadVariableOp¿
actor/dense/MatMulMatMul-actor/batch_normalization/batchnorm/add_1:z:0)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/MatMul±
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp²
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/ReluÂ
:actor/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:actor/batch_normalization_1/moments/mean/reduction_indicesü
(actor/batch_normalization_1/moments/meanMeanactor/dense/Relu:activations:0Cactor/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2*
(actor/batch_normalization_1/moments/meanÑ
0actor/batch_normalization_1/moments/StopGradientStopGradient1actor/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	22
0actor/batch_normalization_1/moments/StopGradient
5actor/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceactor/dense/Relu:activations:09actor/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5actor/batch_normalization_1/moments/SquaredDifferenceÊ
>actor/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>actor/batch_normalization_1/moments/variance/reduction_indices£
,actor/batch_normalization_1/moments/varianceMean9actor/batch_normalization_1/moments/SquaredDifference:z:0Gactor/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2.
,actor/batch_normalization_1/moments/varianceÕ
+actor/batch_normalization_1/moments/SqueezeSqueeze1actor/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2-
+actor/batch_normalization_1/moments/SqueezeÝ
-actor/batch_normalization_1/moments/Squeeze_1Squeeze5actor/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2/
-actor/batch_normalization_1/moments/Squeeze_1«
1actor/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization_1/AssignMovingAvg/decayù
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpCactor_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02<
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp
/actor/batch_normalization_1/AssignMovingAvg/subSubBactor/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:04actor/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_1/AssignMovingAvg/sub
/actor/batch_normalization_1/AssignMovingAvg/mulMul3actor/batch_normalization_1/AssignMovingAvg/sub:z:0:actor/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_1/AssignMovingAvg/mulË
+actor/batch_normalization_1/AssignMovingAvgAssignSubVariableOpCactor_batch_normalization_1_assignmovingavg_readvariableop_resource3actor/batch_normalization_1/AssignMovingAvg/mul:z:0;^actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization_1/AssignMovingAvg¯
3actor/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<25
3actor/batch_normalization_1/AssignMovingAvg_1/decayÿ
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpEactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02>
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp
1actor/batch_normalization_1/AssignMovingAvg_1/subSubDactor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:06actor/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_1/AssignMovingAvg_1/sub
1actor/batch_normalization_1/AssignMovingAvg_1/mulMul5actor/batch_normalization_1/AssignMovingAvg_1/sub:z:0<actor/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_1/AssignMovingAvg_1/mulÕ
-actor/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpEactor_batch_normalization_1_assignmovingavg_1_readvariableop_resource5actor/batch_normalization_1/AssignMovingAvg_1/mul:z:0=^actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02/
-actor/batch_normalization_1/AssignMovingAvg_1
+actor/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_1/batchnorm/add/yó
)actor/batch_normalization_1/batchnorm/addAddV26actor/batch_normalization_1/moments/Squeeze_1:output:04actor/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/add¸
+actor/batch_normalization_1/batchnorm/RsqrtRsqrt-actor/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/Rsqrtó
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_1/batchnorm/mulMul/actor/batch_normalization_1/batchnorm/Rsqrt:y:0@actor/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/mulã
+actor/batch_normalization_1/batchnorm/mul_1Mulactor/dense/Relu:activations:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/mul_1ì
+actor/batch_normalization_1/batchnorm/mul_2Mul4actor/batch_normalization_1/moments/Squeeze:output:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/mul_2ç
4actor/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_1/batchnorm/ReadVariableOpò
)actor/batch_normalization_1/batchnorm/subSub<actor/batch_normalization_1/batchnorm/ReadVariableOp:value:0/actor/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/subö
+actor/batch_normalization_1/batchnorm/add_1AddV2/actor/batch_normalization_1/batchnorm/mul_1:z:0-actor/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/add_1¹
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#actor/dense_1/MatMul/ReadVariableOpÇ
actor/dense_1/MatMulMatMul/actor/batch_normalization_1/batchnorm/add_1:z:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/MatMul·
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOpº
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/BiasAdd
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/ReluÂ
:actor/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:actor/batch_normalization_2/moments/mean/reduction_indicesþ
(actor/batch_normalization_2/moments/meanMean actor/dense_1/Relu:activations:0Cactor/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2*
(actor/batch_normalization_2/moments/meanÑ
0actor/batch_normalization_2/moments/StopGradientStopGradient1actor/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	22
0actor/batch_normalization_2/moments/StopGradient
5actor/batch_normalization_2/moments/SquaredDifferenceSquaredDifference actor/dense_1/Relu:activations:09actor/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5actor/batch_normalization_2/moments/SquaredDifferenceÊ
>actor/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>actor/batch_normalization_2/moments/variance/reduction_indices£
,actor/batch_normalization_2/moments/varianceMean9actor/batch_normalization_2/moments/SquaredDifference:z:0Gactor/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2.
,actor/batch_normalization_2/moments/varianceÕ
+actor/batch_normalization_2/moments/SqueezeSqueeze1actor/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2-
+actor/batch_normalization_2/moments/SqueezeÝ
-actor/batch_normalization_2/moments/Squeeze_1Squeeze5actor/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2/
-actor/batch_normalization_2/moments/Squeeze_1«
1actor/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1actor/batch_normalization_2/AssignMovingAvg/decayù
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpCactor_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02<
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp
/actor/batch_normalization_2/AssignMovingAvg/subSubBactor/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:04actor/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_2/AssignMovingAvg/sub
/actor/batch_normalization_2/AssignMovingAvg/mulMul3actor/batch_normalization_2/AssignMovingAvg/sub:z:0:actor/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:21
/actor/batch_normalization_2/AssignMovingAvg/mulË
+actor/batch_normalization_2/AssignMovingAvgAssignSubVariableOpCactor_batch_normalization_2_assignmovingavg_readvariableop_resource3actor/batch_normalization_2/AssignMovingAvg/mul:z:0;^actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02-
+actor/batch_normalization_2/AssignMovingAvg¯
3actor/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<25
3actor/batch_normalization_2/AssignMovingAvg_1/decayÿ
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpEactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02>
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp
1actor/batch_normalization_2/AssignMovingAvg_1/subSubDactor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:06actor/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_2/AssignMovingAvg_1/sub
1actor/batch_normalization_2/AssignMovingAvg_1/mulMul5actor/batch_normalization_2/AssignMovingAvg_1/sub:z:0<actor/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:23
1actor/batch_normalization_2/AssignMovingAvg_1/mulÕ
-actor/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpEactor_batch_normalization_2_assignmovingavg_1_readvariableop_resource5actor/batch_normalization_2/AssignMovingAvg_1/mul:z:0=^actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02/
-actor/batch_normalization_2/AssignMovingAvg_1
+actor/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_2/batchnorm/add/yó
)actor/batch_normalization_2/batchnorm/addAddV26actor/batch_normalization_2/moments/Squeeze_1:output:04actor/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/add¸
+actor/batch_normalization_2/batchnorm/RsqrtRsqrt-actor/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/Rsqrtó
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_2/batchnorm/mulMul/actor/batch_normalization_2/batchnorm/Rsqrt:y:0@actor/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/mulå
+actor/batch_normalization_2/batchnorm/mul_1Mul actor/dense_1/Relu:activations:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/mul_1ì
+actor/batch_normalization_2/batchnorm/mul_2Mul4actor/batch_normalization_2/moments/Squeeze:output:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/mul_2ç
4actor/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_2/batchnorm/ReadVariableOpò
)actor/batch_normalization_2/batchnorm/subSub<actor/batch_normalization_2/batchnorm/ReadVariableOp:value:0/actor/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/subö
+actor/batch_normalization_2/batchnorm/add_1AddV2/actor/batch_normalization_2/batchnorm/mul_1:z:0-actor/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/add_1
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
mean/MatMul/ReadVariableOp«
mean/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/MatMul
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/BiasAdd²
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!log_std_dev/MatMul/ReadVariableOpÀ
log_std_dev/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/MatMul°
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp±
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y«
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2À

NoOpNoOp*^actor/batch_normalization/AssignMovingAvg9^actor/batch_normalization/AssignMovingAvg/ReadVariableOp,^actor/batch_normalization/AssignMovingAvg_1;^actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^actor/batch_normalization/batchnorm/ReadVariableOp7^actor/batch_normalization/batchnorm/mul/ReadVariableOp,^actor/batch_normalization_1/AssignMovingAvg;^actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp.^actor/batch_normalization_1/AssignMovingAvg_1=^actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^actor/batch_normalization_1/batchnorm/ReadVariableOp9^actor/batch_normalization_1/batchnorm/mul/ReadVariableOp,^actor/batch_normalization_2/AssignMovingAvg;^actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp.^actor/batch_normalization_2/AssignMovingAvg_1=^actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^actor/batch_normalization_2/batchnorm/ReadVariableOp9^actor/batch_normalization_2/batchnorm/mul/ReadVariableOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2V
)actor/batch_normalization/AssignMovingAvg)actor/batch_normalization/AssignMovingAvg2t
8actor/batch_normalization/AssignMovingAvg/ReadVariableOp8actor/batch_normalization/AssignMovingAvg/ReadVariableOp2Z
+actor/batch_normalization/AssignMovingAvg_1+actor/batch_normalization/AssignMovingAvg_12x
:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp:actor/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2actor/batch_normalization/batchnorm/ReadVariableOp2actor/batch_normalization/batchnorm/ReadVariableOp2p
6actor/batch_normalization/batchnorm/mul/ReadVariableOp6actor/batch_normalization/batchnorm/mul/ReadVariableOp2Z
+actor/batch_normalization_1/AssignMovingAvg+actor/batch_normalization_1/AssignMovingAvg2x
:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp:actor/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^
-actor/batch_normalization_1/AssignMovingAvg_1-actor/batch_normalization_1/AssignMovingAvg_12|
<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp<actor/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4actor/batch_normalization_1/batchnorm/ReadVariableOp4actor/batch_normalization_1/batchnorm/ReadVariableOp2t
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp2Z
+actor/batch_normalization_2/AssignMovingAvg+actor/batch_normalization_2/AssignMovingAvg2x
:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp:actor/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^
-actor/batch_normalization_2/AssignMovingAvg_1-actor/batch_normalization_2/AssignMovingAvg_12|
<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp<actor/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4actor/batch_normalization_2/batchnorm/ReadVariableOp4actor/batch_normalization_2/batchnorm/ReadVariableOp2t
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ð

'__inference_mean_layer_call_fn_26719358

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_267176862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ö
C__inference_dense_layer_call_and_return_conditional_losses_26719468

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸A
ð
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26717757
x
actor_26717643:
actor_26717645:
actor_26717647:
actor_26717649:!
actor_26717651:	
actor_26717653:	
actor_26717655:	
actor_26717657:	
actor_26717659:	
actor_26717661:	"
actor_26717663:

actor_26717665:	
actor_26717667:	
actor_26717669:	
actor_26717671:	
actor_26717673:	 
mean_26717687:	
mean_26717689:'
log_std_dev_26717703:	"
log_std_dev_26717705:
identity

identity_1

identity_2¢actor/StatefulPartitionedCall¢#log_std_dev/StatefulPartitionedCall¢mean/StatefulPartitionedCall
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_26717643actor_26717645actor_26717647actor_26717649actor_26717651actor_26717653actor_26717655actor_26717657actor_26717659actor_26717661actor_26717663actor_26717665actor_26717667actor_26717669actor_26717671actor_26717673*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267173472
actor/StatefulPartitionedCall¦
mean/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0mean_26717687mean_26717689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_267176862
mean/StatefulPartitionedCallÉ
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0log_std_dev_26717703log_std_dev_26717705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_267177022%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y»
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulu
addAddV2%mean/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adds
subSubadd:z:0%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3m
TanhTanh%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2³
NoOpNoOp^actor/StatefulPartitionedCall$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ê*
ð
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719631

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²A
ð
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26717970
x
actor_26717878:
actor_26717880:
actor_26717882:
actor_26717884:!
actor_26717886:	
actor_26717888:	
actor_26717890:	
actor_26717892:	
actor_26717894:	
actor_26717896:	"
actor_26717898:

actor_26717900:	
actor_26717902:	
actor_26717904:	
actor_26717906:	
actor_26717908:	 
mean_26717911:	
mean_26717913:'
log_std_dev_26717916:	"
log_std_dev_26717918:
identity

identity_1

identity_2¢actor/StatefulPartitionedCall¢#log_std_dev/StatefulPartitionedCall¢mean/StatefulPartitionedCallý
actor/StatefulPartitionedCallStatefulPartitionedCallxactor_26717878actor_26717880actor_26717882actor_26717884actor_26717886actor_26717888actor_26717890actor_26717892actor_26717894actor_26717896actor_26717898actor_26717900actor_26717902actor_26717904actor_26717906actor_26717908*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267174822
actor/StatefulPartitionedCall¦
mean/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0mean_26717911mean_26717913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_267176862
mean/StatefulPartitionedCallÉ
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall&actor/StatefulPartitionedCall:output:0log_std_dev_26717916log_std_dev_26717918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_267177022%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y»
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulu
addAddV2%mean/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adds
subSubadd:z:0%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3m
TanhTanh%mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2³
NoOpNoOp^actor/StatefulPartitionedCall$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
X
©
$__inference__traced_restore_26719812
file_prefix/
assignvariableop_mean_kernel:	*
assignvariableop_1_mean_bias:8
%assignvariableop_2_log_std_dev_kernel:	1
#assignvariableop_3_log_std_dev_bias::
,assignvariableop_4_batch_normalization_gamma:9
+assignvariableop_5_batch_normalization_beta:2
assignvariableop_6_dense_kernel:	,
assignvariableop_7_dense_bias:	=
.assignvariableop_8_batch_normalization_1_gamma:	<
-assignvariableop_9_batch_normalization_1_beta:	6
"assignvariableop_10_dense_1_kernel:
/
 assignvariableop_11_dense_1_bias:	>
/assignvariableop_12_batch_normalization_2_gamma:	=
.assignvariableop_13_batch_normalization_2_beta:	A
3assignvariableop_14_batch_normalization_moving_mean:E
7assignvariableop_15_batch_normalization_moving_variance:D
5assignvariableop_16_batch_normalization_1_moving_mean:	H
9assignvariableop_17_batch_normalization_1_moving_variance:	D
5assignvariableop_18_batch_normalization_2_moving_mean:	H
9assignvariableop_19_batch_normalization_2_moving_variance:	
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¯
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*»
value±B®B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_mean_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¡
AssignVariableOp_1AssignVariableOpassignvariableop_1_mean_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ª
AssignVariableOp_2AssignVariableOp%assignvariableop_2_log_std_dev_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¨
AssignVariableOp_3AssignVariableOp#assignvariableop_3_log_std_dev_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14»
AssignVariableOp_14AssignVariableOp3assignvariableop_14_batch_normalization_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¿
AssignVariableOp_15AssignVariableOp7assignvariableop_15_batch_normalization_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16½
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Á
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18½
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Á
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20f
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_21þ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
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
ê*
ð
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26717038

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26717140

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26716978

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó

(__inference_dense_layer_call_fn_26719477

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_267173052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

(__inference_actor_layer_call_fn_26717382
batch_normalization_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	
identity¢StatefulPartitionedCallÂ
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
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267173472
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namebatch_normalization_input

¶
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719597

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
Ñ
6__inference_batch_normalization_layer_call_fn_26719457

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ù
E__inference_dense_1_layer_call_and_return_conditional_losses_26719568

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

*__inference_dense_1_layer_call_fn_26719577

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_267173312
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

&__inference_signature_wrapper_26718307
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:	

unknown_18:
identity

identity_1

identity_2¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_267167922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
þ

.__inference_log_std_dev_layer_call_fn_26719377

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_267177022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
£
5__inference_private__mlp_actor_layer_call_fn_26719042
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:	

unknown_18:
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_267179702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¯

û
I__inference_log_std_dev_layer_call_and_return_conditional_losses_26717702

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º$
ã
C__inference_actor_layer_call_and_return_conditional_losses_26717482

inputs*
batch_normalization_26717444:*
batch_normalization_26717446:*
batch_normalization_26717448:*
batch_normalization_26717450:!
dense_26717453:	
dense_26717455:	-
batch_normalization_1_26717458:	-
batch_normalization_1_26717460:	-
batch_normalization_1_26717462:	-
batch_normalization_1_26717464:	$
dense_1_26717467:

dense_1_26717469:	-
batch_normalization_2_26717472:	-
batch_normalization_2_26717474:	-
batch_normalization_2_26717476:	-
batch_normalization_2_26717478:	
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_26717444batch_normalization_26717446batch_normalization_26717448batch_normalization_26717450*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168762-
+batch_normalization/StatefulPartitionedCallº
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26717453dense_26717455*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_267173052
dense/StatefulPartitionedCall¾
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_26717458batch_normalization_1_26717460batch_normalization_1_26717462batch_normalization_1_26717464*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267170382/
-batch_normalization_1/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_26717467dense_1_26717469*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_267173312!
dense_1/StatefulPartitionedCallÀ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_26717472batch_normalization_2_26717474batch_normalization_2_26717476batch_normalization_2_26717478*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267172002/
-batch_normalization_2/StatefulPartitionedCall
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é®
ã
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718727
input_1I
;actor_batch_normalization_batchnorm_readvariableop_resource:M
?actor_batch_normalization_batchnorm_mul_readvariableop_resource:K
=actor_batch_normalization_batchnorm_readvariableop_1_resource:K
=actor_batch_normalization_batchnorm_readvariableop_2_resource:=
*actor_dense_matmul_readvariableop_resource:	:
+actor_dense_biasadd_readvariableop_resource:	L
=actor_batch_normalization_1_batchnorm_readvariableop_resource:	P
Aactor_batch_normalization_1_batchnorm_mul_readvariableop_resource:	N
?actor_batch_normalization_1_batchnorm_readvariableop_1_resource:	N
?actor_batch_normalization_1_batchnorm_readvariableop_2_resource:	@
,actor_dense_1_matmul_readvariableop_resource:
<
-actor_dense_1_biasadd_readvariableop_resource:	L
=actor_batch_normalization_2_batchnorm_readvariableop_resource:	P
Aactor_batch_normalization_2_batchnorm_mul_readvariableop_resource:	N
?actor_batch_normalization_2_batchnorm_readvariableop_1_resource:	N
?actor_batch_normalization_2_batchnorm_readvariableop_2_resource:	6
#mean_matmul_readvariableop_resource:	2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢2actor/batch_normalization/batchnorm/ReadVariableOp¢4actor/batch_normalization/batchnorm/ReadVariableOp_1¢4actor/batch_normalization/batchnorm/ReadVariableOp_2¢6actor/batch_normalization/batchnorm/mul/ReadVariableOp¢4actor/batch_normalization_1/batchnorm/ReadVariableOp¢6actor/batch_normalization_1/batchnorm/ReadVariableOp_1¢6actor/batch_normalization_1/batchnorm/ReadVariableOp_2¢8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp¢4actor/batch_normalization_2/batchnorm/ReadVariableOp¢6actor/batch_normalization_2/batchnorm/ReadVariableOp_1¢6actor/batch_normalization_2/batchnorm/ReadVariableOp_2¢8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"actor/dense/BiasAdd/ReadVariableOp¢!actor/dense/MatMul/ReadVariableOp¢$actor/dense_1/BiasAdd/ReadVariableOp¢#actor/dense_1/MatMul/ReadVariableOp¢"log_std_dev/BiasAdd/ReadVariableOp¢!log_std_dev/MatMul/ReadVariableOp¢mean/BiasAdd/ReadVariableOp¢mean/MatMul/ReadVariableOpà
2actor/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;actor_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype024
2actor/batch_normalization/batchnorm/ReadVariableOp
)actor/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)actor/batch_normalization/batchnorm/add/yð
'actor/batch_normalization/batchnorm/addAddV2:actor/batch_normalization/batchnorm/ReadVariableOp:value:02actor/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/add±
)actor/batch_normalization/batchnorm/RsqrtRsqrt+actor/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/Rsqrtì
6actor/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?actor_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype028
6actor/batch_normalization/batchnorm/mul/ReadVariableOpí
'actor/batch_normalization/batchnorm/mulMul-actor/batch_normalization/batchnorm/Rsqrt:y:0>actor/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/mulÅ
)actor/batch_normalization/batchnorm/mul_1Mulinput_1+actor/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/mul_1æ
4actor/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=actor_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype026
4actor/batch_normalization/batchnorm/ReadVariableOp_1í
)actor/batch_normalization/batchnorm/mul_2Mul<actor/batch_normalization/batchnorm/ReadVariableOp_1:value:0+actor/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2+
)actor/batch_normalization/batchnorm/mul_2æ
4actor/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=actor_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype026
4actor/batch_normalization/batchnorm/ReadVariableOp_2ë
'actor/batch_normalization/batchnorm/subSub<actor/batch_normalization/batchnorm/ReadVariableOp_2:value:0-actor/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2)
'actor/batch_normalization/batchnorm/subí
)actor/batch_normalization/batchnorm/add_1AddV2-actor/batch_normalization/batchnorm/mul_1:z:0+actor/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)actor/batch_normalization/batchnorm/add_1²
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!actor/dense/MatMul/ReadVariableOp¿
actor/dense/MatMulMatMul-actor/batch_normalization/batchnorm/add_1:z:0)actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/MatMul±
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp²
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/BiasAdd}
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense/Reluç
4actor/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_1/batchnorm/ReadVariableOp
+actor/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_1/batchnorm/add/yù
)actor/batch_normalization_1/batchnorm/addAddV2<actor/batch_normalization_1/batchnorm/ReadVariableOp:value:04actor/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/add¸
+actor/batch_normalization_1/batchnorm/RsqrtRsqrt-actor/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/Rsqrtó
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_1/batchnorm/mulMul/actor/batch_normalization_1/batchnorm/Rsqrt:y:0@actor/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/mulã
+actor/batch_normalization_1/batchnorm/mul_1Mulactor/dense/Relu:activations:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/mul_1í
6actor/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?actor_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_1/batchnorm/ReadVariableOp_1ö
+actor/batch_normalization_1/batchnorm/mul_2Mul>actor/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-actor/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_1/batchnorm/mul_2í
6actor/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?actor_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_1/batchnorm/ReadVariableOp_2ô
)actor/batch_normalization_1/batchnorm/subSub>actor/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/actor/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_1/batchnorm/subö
+actor/batch_normalization_1/batchnorm/add_1AddV2/actor/batch_normalization_1/batchnorm/mul_1:z:0-actor/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_1/batchnorm/add_1¹
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#actor/dense_1/MatMul/ReadVariableOpÇ
actor/dense_1/MatMulMatMul/actor/batch_normalization_1/batchnorm/add_1:z:0+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/MatMul·
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOpº
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/BiasAdd
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense_1/Reluç
4actor/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=actor_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype026
4actor/batch_normalization_2/batchnorm/ReadVariableOp
+actor/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+actor/batch_normalization_2/batchnorm/add/yù
)actor/batch_normalization_2/batchnorm/addAddV2<actor/batch_normalization_2/batchnorm/ReadVariableOp:value:04actor/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/add¸
+actor/batch_normalization_2/batchnorm/RsqrtRsqrt-actor/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/Rsqrtó
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAactor_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02:
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOpö
)actor/batch_normalization_2/batchnorm/mulMul/actor/batch_normalization_2/batchnorm/Rsqrt:y:0@actor/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/mulå
+actor/batch_normalization_2/batchnorm/mul_1Mul actor/dense_1/Relu:activations:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/mul_1í
6actor/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?actor_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_2/batchnorm/ReadVariableOp_1ö
+actor/batch_normalization_2/batchnorm/mul_2Mul>actor/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-actor/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2-
+actor/batch_normalization_2/batchnorm/mul_2í
6actor/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?actor_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype028
6actor/batch_normalization_2/batchnorm/ReadVariableOp_2ô
)actor/batch_normalization_2/batchnorm/subSub>actor/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/actor/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2+
)actor/batch_normalization_2/batchnorm/subö
+actor/batch_normalization_2/batchnorm/add_1AddV2/actor/batch_normalization_2/batchnorm/mul_1:z:0-actor/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+actor/batch_normalization_2/batchnorm/add_1
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
mean/MatMul/ReadVariableOp«
mean/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/MatMul
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mean/BiasAdd²
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!log_std_dev/MatMul/ReadVariableOpÀ
log_std_dev/MatMulMatMul/actor/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/MatMul°
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp±
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y«
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueV
ExpExpclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *  ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
addAddV2mean/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
subSubadd:z:0mean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subZ
Exp_1Expclip_by_value:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22	
add_1/yf
add_1AddV2	Exp_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
truedivRealDivsub:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_2AddV2pow:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?2	
add_3/yf
add_3AddV2	add_2:z:0add_3/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_2/xd
mul_2Mulmul_2/x:output:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   À2	
mul_3/xb
mul_3Mulmul_3/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3]
SoftplusSoftplus	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusj
sub_2Sub	sub_1:z:0Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
Sum_1a
sub_3SubSum:output:0Sum_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_3]
TanhTanhmean/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhS
Tanh_1Tanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_5/yc
mul_5MulTanh:y:0mul_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_6/ye
mul_6Mul
Tanh_1:y:0mul_6/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6d
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1d

Identity_2Identity	sub_3:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp3^actor/batch_normalization/batchnorm/ReadVariableOp5^actor/batch_normalization/batchnorm/ReadVariableOp_15^actor/batch_normalization/batchnorm/ReadVariableOp_27^actor/batch_normalization/batchnorm/mul/ReadVariableOp5^actor/batch_normalization_1/batchnorm/ReadVariableOp7^actor/batch_normalization_1/batchnorm/ReadVariableOp_17^actor/batch_normalization_1/batchnorm/ReadVariableOp_29^actor/batch_normalization_1/batchnorm/mul/ReadVariableOp5^actor/batch_normalization_2/batchnorm/ReadVariableOp7^actor/batch_normalization_2/batchnorm/ReadVariableOp_17^actor/batch_normalization_2/batchnorm/ReadVariableOp_29^actor/batch_normalization_2/batchnorm/mul/ReadVariableOp#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2h
2actor/batch_normalization/batchnorm/ReadVariableOp2actor/batch_normalization/batchnorm/ReadVariableOp2l
4actor/batch_normalization/batchnorm/ReadVariableOp_14actor/batch_normalization/batchnorm/ReadVariableOp_12l
4actor/batch_normalization/batchnorm/ReadVariableOp_24actor/batch_normalization/batchnorm/ReadVariableOp_22p
6actor/batch_normalization/batchnorm/mul/ReadVariableOp6actor/batch_normalization/batchnorm/mul/ReadVariableOp2l
4actor/batch_normalization_1/batchnorm/ReadVariableOp4actor/batch_normalization_1/batchnorm/ReadVariableOp2p
6actor/batch_normalization_1/batchnorm/ReadVariableOp_16actor/batch_normalization_1/batchnorm/ReadVariableOp_12p
6actor/batch_normalization_1/batchnorm/ReadVariableOp_26actor/batch_normalization_1/batchnorm/ReadVariableOp_22t
8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp8actor/batch_normalization_1/batchnorm/mul/ReadVariableOp2l
4actor/batch_normalization_2/batchnorm/ReadVariableOp4actor/batch_normalization_2/batchnorm/ReadVariableOp2p
6actor/batch_normalization_2/batchnorm/ReadVariableOp_16actor/batch_normalization_2/batchnorm/ReadVariableOp_12p
6actor/batch_normalization_2/batchnorm/ReadVariableOp_26actor/batch_normalization_2/batchnorm/ReadVariableOp_22t
8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp8actor/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

©
5__inference_private__mlp_actor_layer_call_fn_26718944
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:	

unknown_18:
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_267177572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
óÇ
¹
C__inference_actor_layer_call_and_return_conditional_losses_26719265

inputsI
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_1_batchnorm_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_2_batchnorm_readvariableop_resource:	
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢%batch_normalization_1/AssignMovingAvg¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢'batch_normalization_1/AssignMovingAvg_1¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢%batch_normalization_2/AssignMovingAvg¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢'batch_normalization_2/AssignMovingAvg_1¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp²
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesË
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2"
 batch_normalization/moments/mean¸
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:2*
(batch_normalization/moments/StopGradientà
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-batch_normalization/moments/SquaredDifferenceº
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization/moments/variance¼
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/SqueezeÄ
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2+
)batch_normalization/AssignMovingAvg/decayà
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpè
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/subß
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul£
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization/AssignMovingAvg_1/decayæ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpð
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/subç
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mul­
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yÒ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul²
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/mul_1Ë
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2Î
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpÑ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/add_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp§
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¶
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indicesä
"batch_normalization_1/moments/meanMeandense/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_1/moments/mean¿
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_1/moments/StopGradientù
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_1/moments/SquaredDifference¾
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2(
&batch_normalization_1/moments/varianceÃ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeË
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_1/AssignMovingAvg/decayç
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpñ
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_1/AssignMovingAvg/subè
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_1/AssignMovingAvg/mul­
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg£
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_1/AssignMovingAvg_1/decayí
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpù
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_1/AssignMovingAvg_1/subð
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_1/AssignMovingAvg_1/mul·
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yÛ
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_1/batchnorm/Rsqrtá
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/mulË
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_1/batchnorm/mul_1Ô
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_1/batchnorm/mul_2Õ
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpÚ
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_1/batchnorm/add_1§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¯
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu¶
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesæ
"batch_normalization_2/moments/meanMeandense_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_2/moments/mean¿
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_2/moments/StopGradientû
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_2/moments/SquaredDifference¾
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2(
&batch_normalization_2/moments/varianceÃ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeË
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_2/AssignMovingAvg/decayç
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpñ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_2/AssignMovingAvg/subè
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_2/AssignMovingAvg/mul­
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg£
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_2/AssignMovingAvg_1/decayí
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpù
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_2/AssignMovingAvg_1/subð
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_2/AssignMovingAvg_1/mul·
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yÛ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/add¦
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_2/batchnorm/Rsqrtá
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/mulÍ
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_2/batchnorm/mul_1Ô
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_2/batchnorm/mul_2Õ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpÚ
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_2/batchnorm/subÞ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_2/batchnorm/add_1
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
£
5__inference_private__mlp_actor_layer_call_fn_26718993
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:	

unknown_18:
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_267177572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
á
×
8__inference_batch_normalization_1_layer_call_fn_26719557

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267170382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù$
ö
C__inference_actor_layer_call_and_return_conditional_losses_26717595
batch_normalization_input*
batch_normalization_26717557:*
batch_normalization_26717559:*
batch_normalization_26717561:*
batch_normalization_26717563:!
dense_26717566:	
dense_26717568:	-
batch_normalization_1_26717571:	-
batch_normalization_1_26717573:	-
batch_normalization_1_26717575:	-
batch_normalization_1_26717577:	$
dense_1_26717580:

dense_1_26717582:	-
batch_normalization_2_26717585:	-
batch_normalization_2_26717587:	-
batch_normalization_2_26717589:	-
batch_normalization_2_26717591:	
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_26717557batch_normalization_26717559batch_normalization_26717561batch_normalization_26717563*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168162-
+batch_normalization/StatefulPartitionedCallº
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26717566dense_26717568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_267173052
dense/StatefulPartitionedCallÀ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_26717571batch_normalization_1_26717573batch_normalization_1_26717575batch_normalization_1_26717577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267169782/
-batch_normalization_1/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_26717580dense_1_26717582*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_267173312!
dense_1/StatefulPartitionedCallÂ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_26717585batch_normalization_2_26717587batch_normalization_2_26717589batch_normalization_2_26717591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267171402/
-batch_normalization_2/StatefulPartitionedCall
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namebatch_normalization_input
×
Ñ
6__inference_batch_normalization_layer_call_fn_26719444

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168162
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶

(__inference_actor_layer_call_fn_26719339

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	
identity¢StatefulPartitionedCall©
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_267174822
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ö
C__inference_dense_layer_call_and_return_conditional_losses_26717305

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
×
8__inference_batch_normalization_1_layer_call_fn_26719544

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267169782
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì*
ê
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26716876

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó$
ö
C__inference_actor_layer_call_and_return_conditional_losses_26717636
batch_normalization_input*
batch_normalization_26717598:*
batch_normalization_26717600:*
batch_normalization_26717602:*
batch_normalization_26717604:!
dense_26717607:	
dense_26717609:	-
batch_normalization_1_26717612:	-
batch_normalization_1_26717614:	-
batch_normalization_1_26717616:	-
batch_normalization_1_26717618:	$
dense_1_26717621:

dense_1_26717623:	-
batch_normalization_2_26717626:	-
batch_normalization_2_26717628:	-
batch_normalization_2_26717630:	-
batch_normalization_2_26717632:	
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_26717598batch_normalization_26717600batch_normalization_26717602batch_normalization_26717604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_267168762-
+batch_normalization/StatefulPartitionedCallº
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_26717607dense_26717609*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_267173052
dense/StatefulPartitionedCall¾
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_26717612batch_normalization_1_26717614batch_normalization_1_26717616batch_normalization_1_26717618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_267170382/
-batch_normalization_1/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_26717621dense_1_26717623*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_267173312!
dense_1/StatefulPartitionedCallÀ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_26717626batch_normalization_2_26717628batch_normalization_2_26717630batch_normalization_2_26717632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267172002/
-batch_normalization_2/StatefulPartitionedCall
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namebatch_normalization_input
õ
°
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719397

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ù
E__inference_dense_1_layer_call_and_return_conditional_losses_26717331

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì*
ê
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719431

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
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
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

û
I__inference_log_std_dev_layer_call_and_return_conditional_losses_26719368

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
×
8__inference_batch_normalization_2_layer_call_fn_26719644

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_267171402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*£
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ8
output_3,
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:û­
ô
	_body
_mu
_log_std
trainable_variables
	variables
regularization_losses
	keras_api

signatures
*r&call_and_return_all_conditional_losses
s__call__
t_default_save_signature"
_tf_keras_model
í
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
*u&call_and_return_all_conditional_losses
v__call__"
_tf_keras_sequential
»

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"
_tf_keras_layer
»

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"
_tf_keras_layer

0
1
 2
!3
"4
#5
$6
%7
&8
'9
10
11
12
13"
trackable_list_wrapper
¶
0
1
(2
)3
 4
!5
"6
#7
*8
+9
$10
%11
&12
'13
,14
-15
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
trainable_variables
.layer_regularization_losses
/layer_metrics
0non_trainable_variables
	variables
regularization_losses
1metrics

2layers
s__call__
t_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
ê
3axis
	gamma
beta
(moving_mean
)moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
*|&call_and_return_all_conditional_losses
}__call__"
_tf_keras_layer
»

 kernel
!bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
*~&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
<axis
	"gamma
#beta
*moving_mean
+moving_variance
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

$kernel
%bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Eaxis
	&gamma
'beta
,moving_mean
-moving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9"
trackable_list_wrapper

0
1
(2
)3
 4
!5
"6
#7
*8
+9
$10
%11
&12
'13
,14
-15"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
Jlayer_regularization_losses
Klayer_metrics
Lnon_trainable_variables
	variables
regularization_losses
Mmetrics

Nlayers
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
:	2mean/kernel
:2	mean/bias
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
­
trainable_variables
Olayer_regularization_losses
Player_metrics
Qnon_trainable_variables
	variables
regularization_losses
Rmetrics

Slayers
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
%:#	2log_std_dev/kernel
:2log_std_dev/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
Vnon_trainable_variables
	variables
regularization_losses
Wmetrics

Xlayers
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
':%2batch_normalization/gamma
&:$2batch_normalization/beta
:	2dense/kernel
:2
dense/bias
*:(2batch_normalization_1/gamma
):'2batch_normalization_1/beta
": 
2dense_1/kernel
:2dense_1/bias
*:(2batch_normalization_2/gamma
):'2batch_normalization_2/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
2:0 (2!batch_normalization_1/moving_mean
6:4 (2%batch_normalization_1/moving_variance
2:0 (2!batch_normalization_2/moving_mean
6:4 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
4trainable_variables
Ylayer_regularization_losses
Zlayer_metrics
[non_trainable_variables
5	variables
6regularization_losses
\metrics

]layers
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8trainable_variables
^layer_regularization_losses
_layer_metrics
`non_trainable_variables
9	variables
:regularization_losses
ametrics

blayers
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
<
"0
#1
*2
+3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
=trainable_variables
clayer_regularization_losses
dlayer_metrics
enon_trainable_variables
>	variables
?regularization_losses
fmetrics

glayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Atrainable_variables
hlayer_regularization_losses
ilayer_metrics
jnon_trainable_variables
B	variables
Cregularization_losses
kmetrics

llayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
<
&0
'1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ftrainable_variables
mlayer_regularization_losses
nlayer_metrics
onon_trainable_variables
G	variables
Hregularization_losses
pmetrics

qlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
 "
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
(0
)1"
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
*0
+1"
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
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ü2ù
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718433
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718601
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718727
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718895®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
5__inference_private__mlp_actor_layer_call_fn_26718944
5__inference_private__mlp_actor_layer_call_fn_26718993
5__inference_private__mlp_actor_layer_call_fn_26719042
5__inference_private__mlp_actor_layer_call_fn_26719091®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÎBË
#__inference__wrapped_model_26716792input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
C__inference_actor_layer_call_and_return_conditional_losses_26719157
C__inference_actor_layer_call_and_return_conditional_losses_26719265
C__inference_actor_layer_call_and_return_conditional_losses_26717595
C__inference_actor_layer_call_and_return_conditional_losses_26717636À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
(__inference_actor_layer_call_fn_26717382
(__inference_actor_layer_call_fn_26719302
(__inference_actor_layer_call_fn_26719339
(__inference_actor_layer_call_fn_26717554À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_mean_layer_call_and_return_conditional_losses_26719349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_mean_layer_call_fn_26719358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_log_std_dev_layer_call_and_return_conditional_losses_26719368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_log_std_dev_layer_call_fn_26719377¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_26718307input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719397
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719431´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_layer_call_fn_26719444
6__inference_batch_normalization_layer_call_fn_26719457´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_layer_call_and_return_conditional_losses_26719468¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_layer_call_fn_26719477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ä2á
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719497
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719531´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
®2«
8__inference_batch_normalization_1_layer_call_fn_26719544
8__inference_batch_normalization_1_layer_call_fn_26719557´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_dense_1_layer_call_and_return_conditional_losses_26719568¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_1_layer_call_fn_26719577¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ä2á
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719597
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719631´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
®2«
8__inference_batch_normalization_2_layer_call_fn_26719644
8__inference_batch_normalization_2_layer_call_fn_26719657´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
#__inference__wrapped_model_26716792Û)( !+"*#$%-&,'0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ
*
output_3
output_3ÿÿÿÿÿÿÿÿÿÎ
C__inference_actor_layer_call_and_return_conditional_losses_26717595)( !+"*#$%-&,'J¢G
@¢=
30
batch_normalization_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Î
C__inference_actor_layer_call_and_return_conditional_losses_26717636() !*+"#$%,-&'J¢G
@¢=
30
batch_normalization_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 º
C__inference_actor_layer_call_and_return_conditional_losses_26719157s)( !+"*#$%-&,'7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 º
C__inference_actor_layer_call_and_return_conditional_losses_26719265s() !*+"#$%,-&'7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
(__inference_actor_layer_call_fn_26717382y)( !+"*#$%-&,'J¢G
@¢=
30
batch_normalization_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¥
(__inference_actor_layer_call_fn_26717554y() !*+"#$%,-&'J¢G
@¢=
30
batch_normalization_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_actor_layer_call_fn_26719302f)( !+"*#$%-&,'7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_actor_layer_call_fn_26719339f() !*+"#$%,-&'7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719497d+"*#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26719531d*+"#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_1_layer_call_fn_26719544W+"*#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_1_layer_call_fn_26719557W*+"#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ»
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719597d-&,'4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26719631d,-&'4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_2_layer_call_fn_26719644W-&,'4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_2_layer_call_fn_26719657W,-&'4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719397b)(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_26719431b()3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_layer_call_fn_26719444U)(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_layer_call_fn_26719457U()3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_1_layer_call_and_return_conditional_losses_26719568^$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_1_layer_call_fn_26719577Q$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_layer_call_and_return_conditional_losses_26719468] !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_layer_call_fn_26719477P !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_log_std_dev_layer_call_and_return_conditional_losses_26719368]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_log_std_dev_layer_call_fn_26719377P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_mean_layer_call_and_return_conditional_losses_26719349]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_mean_layer_call_fn_26719358P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718433®)( !+"*#$%-&,'.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c
\¢Y

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718601®() !*+"#$%,-&'.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p
ª "f¢c
\¢Y

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718727´)( !+"*#$%-&,'4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c
\¢Y

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
P__inference_private__mlp_actor_layer_call_and_return_conditional_losses_26718895´() !*+"#$%,-&'4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c
\¢Y

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 Þ
5__inference_private__mlp_actor_layer_call_fn_26718944¤)( !+"*#$%-&,'4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "V¢S

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿØ
5__inference_private__mlp_actor_layer_call_fn_26718993)( !+"*#$%-&,'.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p 
ª "V¢S

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿØ
5__inference_private__mlp_actor_layer_call_fn_26719042() !*+"#$%,-&'.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p
ª "V¢S

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÞ
5__inference_private__mlp_actor_layer_call_fn_26719091¤() !*+"#$%,-&'4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "V¢S

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ
&__inference_signature_wrapper_26718307æ)( !+"*#$%-&,';¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ
*
output_3
output_3ÿÿÿÿÿÿÿÿÿ