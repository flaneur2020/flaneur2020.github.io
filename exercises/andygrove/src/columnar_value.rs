use std::sync::Arc;
use arrow2::array::Array;
use arrow2::datatypes::DataType;

#[derive(Debug, PartialEq)]
pub enum ScalarValue {
    Null,
    Boolean(Option<bool>),
    Int64(Option<i64>),
    Float64(Option<f64>),
    String(Option<String>),
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Boolean(_) => DataType::Boolean,
            Self::Int64(_) => DataType::Int64,
            Self::Float64(_) => DataType::Float64,
            Self::String(_) => DataType::Utf8,
        }
    }

    pub fn to_array_of_size(&self, size: usize) -> ArrayRef {
        panic!("not implemented")
    }
}

type ArrayRef = Arc<dyn Array>;

pub enum ColumnarValue {
    Scalar(ScalarValue),
    Array(ArrayRef),
}

impl ColumnarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Scalar(scalar) => scalar.data_type(),
            Self::Array(array) => array.data_type().clone(),
        }
    }
}