using System;


namespace HelperRoutines
{
    public class OutputConfusionMatrix
    {
        static public void Printstats(in Accord.Statistics.Analysis.ConfusionMatrix cm)
        {
            // Prediction accuracy of training set = 99.51%
            Console.Write(resources.strings.Predictions);
            Printcolor(Math.Round(cm.Accuracy * 100, 2), ConsoleColor.Red, true);
            // Sensitivity(true Positive rate) = 86
            // Specificity(true Negative rate) = 100
            Console.Write("Sensitivity (Recall - true Positive rate) = ");
            Printcolor(Math.Round(cm.Sensitivity * 100, 2), ConsoleColor.Yellow, true);
            Console.Write("Specificity(true Negative rate) = ");
            Printcolor(Math.Round(cm.Specificity * 100, 2), ConsoleColor.Yellow, true);
            Console.Write("Precision (TP / TP + FP) = ");
            Printcolor(Math.Round(Math.Round(cm.Precision * 100, 2)), ConsoleColor.Yellow, true);
            Console.WriteLine("FScore (harmonic mean of Precision and Recall.) = {0}", Math.Round(cm.FScore * 100), 2);

        }

        static private void Printcolor(int value, ConsoleColor color, bool addpercent)
        {
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            if (!addpercent)
                Console.WriteLine(value);
            else
                Console.WriteLine("{0}%", value);
            Console.ForegroundColor = originalColor;
        }
        static private void Printcolor(double value, ConsoleColor color, bool addpercent)
        {
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            if (!addpercent)
                Console.WriteLine(value);
            else
                Console.WriteLine("{0}%", value);
            Console.ForegroundColor = originalColor;
        }

        static private void Printcolor(string value, ConsoleColor color)
        {
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            Console.WriteLine(value);
            Console.ForegroundColor = originalColor;
        }

    }

}