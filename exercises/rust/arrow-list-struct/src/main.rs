use arrow::array::Int16Builder;

fn main() {
    use arrow_array::builder::{ArrayBuilder, ListBuilder, StringBuilder, StructBuilder};
    use arrow_schema::{DataType, Field, Fields};

    // This is an example column that has a List<Struct<List<Struct>>> layout
    let mut example_col = ListBuilder::new(StructBuilder::from_fields(
        vec![
            Field::new("a", DataType::Int16, true),
            Field::new("b", DataType::Int16, true),
        ],
        0,
    ));

    // We can obtain the StructBuilder without issues, because example_col was created with StructBuilder
    let col_struct_builder: &mut StructBuilder = example_col.values();

    for i in 0..10 {
        col_struct_builder
            .field_builder::<Int16Builder>(0)
            .unwrap()
            .append_value(i);
        col_struct_builder
            .field_builder::<Int16Builder>(1)
            .unwrap()
            .append_value(i * 2);
        col_struct_builder.append(true);
    }

    example_col.append(true);

    let array = example_col.finish();

    println!("My array: {:?}", array);
}
