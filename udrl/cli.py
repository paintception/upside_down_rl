import argparse
import dataclasses
import inspect as i
from typing import Callable, Dict, Any
from dataclasses import fields, is_dataclass


def sel_args(kw: Dict[str, Any], fun: Callable) -> Dict[str, Any]:
    """
    Selects keyword arguments relevant to a function.

    Parameters
    ----------
    kw : Dict[str, Any]
        A dictionary of keyword arguments.
    fun : Callable
        The function for which arguments are to be selected.

    Returns
    -------
    Dict[str, Any]
        A new dictionary containing only the keyword arguments
        that are valid parameters for the given function.
    """
    return {
        k: v for k, v in kw.items() if k in list(i.signature(fun).parameters)
    }


def apply(fun: Callable, kw: Dict[str, Any]) -> Any:
    """
    Applies a function with selected keyword arguments.

    Parameters
    ----------
    fun : Callable
        The function to apply.
    kw : Dict[str, Any]
        A dictionary of keyword arguments.

    Returns
    -------
    Any
        The result of calling the function with the selected keyword arguments.
    """
    return fun(**sel_args(kw, fun))


def create_argparse_dict(dataclass_cls):
    """
    Creates an argument parser dictionary configuration from a dataclass.

    This function examines the fields of a dataclass and generates a dictionary
    that can be used to configure an argparse.ArgumentParser.
    It handles boolean fields with special actions, sets default values,
    includes help messages with defaults, and supports optional choices
    and required arguments based on metadata.

    Parameters
    ----------
    dataclass_cls : type
        The dataclass type to create the argument parser dictionary from.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping argument names to dictionaries containing
        argparse configuration options.
    """
    result = {}
    for field in dataclasses.fields(dataclass_cls):
        if not field.init:
            continue
        arg_name = f"--{field.name.replace('_', '-')}"
        if field.type == bool:
            result[arg_name] = dict(
                action=argparse.BooleanOptionalAction,
                default=field.default,
            )
            continue
        result[arg_name] = {
            "type": field.type,
            "default": (
                field.default
                if not dataclasses.is_dataclass(field.type)
                else None
            ),
            "help": f"{field.metadata.get('help', '')}"
            f"(default: {field.default})",
        }
        if choices := field.metadata.get("choices", None):
            result[arg_name]["choices"] = choices
        if required := field.metadata.get("required", None):
            result[arg_name]["required"] = required
    return result


def create_experiment_from_args(
    args: argparse.Namespace, dataclass: Callable[..., Any]
) -> Any:
    """
    Creates an experiment instance from parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        An argparse Namespace object containing parsed command-line arguments.
    dataclass : Callable[..., Any]
        A dataclass constructor that takes keyword arguments corresponding
        to experiment parameters.

    Returns
    -------
    Any
        An instance of the dataclass initialized with the parsed arguments.
    """
    return apply(
        dataclass,
        {
            k.replace("--", "").replace("-", "_"): v
            for k, v in vars(args).items()
        },
    )


def with_meta(default: Any, help: str, **kwargs):
    """
    Creates a dataclass field with default value, help string,
    and additional metadata.

    This function simplifies the creation of dataclass fields by
    providing a convenient way to set a default value,
    a help string, and other metadata attributes for a field.

    Parameters
    ----------
    default : Any
        The default value for the field. If callable,
        it's treated as a default factory.
    help : str
        The help string describing the field's purpose.
    **kwargs
        Additional keyword arguments to be included in the field's metadata.

    Returns
    -------
    dataclasses.Field
        A dataclass Field object with the specified default,
        help, and metadata.
    """
    args: Dict[str, Any] = {"metadata": {"help": help, **kwargs}}
    if callable(default):
        args["default_factory"] = default
    else:
        args["default"] = default
    return dataclasses.field(**args)


def dataclass_non_defaults_to_string(data_obj):
    """Converts non-default values of a dataclass object's fields to a string,
    excluding 'seed' and 'env_name'.

    Parameters
    ----------
    data_obj : dataclass object
        The dataclass object to process.

    Returns
    -------
    str
        A string representation of non-default field values, or "base" if all
        fields have default values
        (excluding the 'seed' and 'env_name' attributes).

    Raises
    ------
    TypeError
        If the input is not a dataclass object.
    """
    if not is_dataclass(data_obj):
        raise TypeError("Input must be a dataclass object.")

    non_defaults = []
    for field in fields(data_obj):
        if field.name == "seed" or field.name == "env_name":
            continue
        if getattr(data_obj, field.name) != field.default:
            non_defaults.append(
                field.name + str(getattr(data_obj, field.name))
            )

    return "_".join(non_defaults) or "base"
