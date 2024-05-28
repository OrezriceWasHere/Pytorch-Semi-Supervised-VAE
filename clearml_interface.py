from clearml import Task
from os import environ as env
import params

ALLOW_CLEARML = True if env.get("ALLOW_CLEARML") == "yes" else False
RUNNING_REMOTE = True if env.get("RUNNING_REMOTE") == "yes" else False


def clearml_init():
    global execution_task
    if ALLOW_CLEARML:

        execution_task = Task.init(project_name="Pytorch Semi Supervised VAE",
                                   task_name="m1 with 10000 epochs, lr = 1e-4",
                                   task_type=Task.TaskTypes.testing,
                                   reuse_last_task_id=False,
                                   )

        if execution_task.running_locally():
            name = input("enter description for task:\n")
            execution_task.set_name(name)

        execution_task.set_parameters_as_dict(params.Params.__dict__)

        if RUNNING_REMOTE:
            execution_task.execute_remotely(queue_name="cpu", exit_process=True)


def clearml_display_image(image, iteration, series, description):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_image(description,

                                                 image=image,
                                                 iteration=iteration,
                                                 series=series)


def add_point_to_graph(title, series, x, y):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_scalar(title, series, value=y, iteration=x)


def add_confusion_matrix(matrix, title, series, iteration):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_confusion_matrix(title, series=series, matrix=matrix, iteration=iteration)


def add_text(text, title, iteration):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_text(title, text, iteration=iteration)
