-- A model labeling function names the adapter that produced its votes, so
-- labeling quality can be attributed to an adapter. Only a model may name one.

ALTER TABLE {{work}}.labeling_function
    ADD COLUMN adapter text REFERENCES {{work}}.adapter (id),
    ADD CONSTRAINT labeling_function_adapter_is_a_model
        CHECK (adapter IS NULL OR kind = 'model');
